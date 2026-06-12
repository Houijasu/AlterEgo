module AlterEgo.Train

// NNUE trainer (M5): float32 training of the (768 -> 256)x2 -> 1 net on
// self-play samples from Datagen, exported quantized for the engine.
//
// Target: t = 0.7 * sigmoid(score/200) + 0.3 * result01   (stm POV)
// Prediction: p = sigmoid(2u) where u = net output ~ cp/400
// Loss: (p - t)^2, Adam, weight clipping to int16-safe range.

open System
open System.IO
open System.Numerics
open System.Threading.Tasks
open AlterEgo.Types

[<Literal>]
let private H = 256      // hidden size (= Nnue.HiddenSize)
[<Literal>]
let private SampleBytes = 100

// ---- sample store: the corpus as 100-byte-aligned slabs ----
// A single byte[] caps at 2GB (~21M samples); slabs remove the ceiling.
// Multi-file corpora: '+'-separated paths, each file's trailing partial
// record dropped INDEPENDENTLY so one short file can't misalign the next.
[<Literal>]
let private SamplesPerSlab = 10_000_000   // 1GB per slab

type private Store =
    { Slabs: byte[][]
      Total: int }

let inline private slabOf (store: Store) (idx: int) = store.Slabs.[idx / SamplesPerSlab]
let inline private offOf (idx: int) = (idx % SamplesPerSlab) * SampleBytes

let private loadStore (spec: string) : Store =
    let paths =
        spec.Split('+')
        |> Array.map (fun p -> p.Trim())
        |> Array.filter (fun p -> p <> "")
    let counts =
        paths
        |> Array.map (fun p ->
            let len = FileInfo(p).Length
            if len % int64 SampleBytes <> 0L then
                printfn "WARNING: %s — %d trailing bytes ignored (truncated mid-record? run scrub)"
                    p (int (len % int64 SampleBytes))
            len / int64 SampleBytes)
    let total = Array.sum counts
    let slabCount = int ((total + int64 SamplesPerSlab - 1L) / int64 SamplesPerSlab)
    let slabs =
        Array.init slabCount (fun i ->
            let samples = min (int64 SamplesPerSlab) (total - int64 i * int64 SamplesPerSlab)
            Array.zeroCreate<byte> (int samples * SampleBytes))
    let mutable next = 0L   // next global sample slot to fill
    for fi in 0 .. paths.Length - 1 do
        use fs = File.OpenRead paths.[fi]
        let mutable remaining = counts.[fi] * int64 SampleBytes
        while remaining > 0L do
            let slab = int (next / int64 SamplesPerSlab)
            let offBytes = int (next % int64 SamplesPerSlab) * SampleBytes
            let want = int (min (int64 (slabs.[slab].Length - offBytes)) remaining)
            let mutable got = 0
            while got < want do
                let r = fs.Read(slabs.[slab], offBytes + got, want - got)
                if r <= 0 then failwithf "unexpected EOF reading %s" paths.[fi]
                got <- got + r
            next <- next + int64 (want / SampleBytes)
            remaining <- remaining - int64 want
    { Slabs = slabs; Total = int total }

// input size is runtime: 768 (flat) / 3072 (own-king) / 6144 (dual-king factorized)
let mutable private In = 768
let mutable private buckets = 1
let mutable private mirror = false
let mutable private dual = false

// WDL auxiliary loss weight. CP loss is sigmoid-space MSE (~0.012 at
// convergence); WDL cross-entropy is in nats (~0.6-1.0) — ~50x larger raw.
// 0.01 keeps the trunk CP-dominated while still training the WDL head.
[<Literal>]
let private LambdaWdl = 0.01f

type private Net =
    { W1: float32[]      // In*H, feature-major
      B1: float32[]
      W2: float32[]      // 2H
      mutable B2: float32
      Wwdl: float32[]    // 3 * 2H: (win,draw,loss) x (stm half | opp half)
      Bwdl: float32[] }  // 3

let private newNet (rng: Random) =
    let scale1 = sqrt (2.0 / 32.0) |> float32      // ~32 active features
    let scale2 = sqrt (2.0 / float (2 * H)) |> float32
    { W1 = Array.init (In * H) (fun _ -> (float32 (rng.NextDouble()) - 0.5f) * 2.0f * scale1)
      B1 = Array.zeroCreate H
      W2 = Array.init (2 * H) (fun _ -> (float32 (rng.NextDouble()) - 0.5f) * 2.0f * scale2)
      B2 = 0.0f
      Wwdl = Array.init (3 * 2 * H) (fun _ -> (float32 (rng.NextDouble()) - 0.5f) * 2.0f * scale2)
      Bwdl = Array.zeroCreate 3 }

type private Grad =
    { GW1: float32[]
      GB1: float32[]
      GW2: float32[]
      mutable GB2: float32
      GWwdl: float32[]
      GBwdl: float32[]
      mutable LossCp: float
      mutable LossWdl: float }

let private newGrad () =
    { GW1 = Array.zeroCreate (In * H)
      GB1 = Array.zeroCreate H
      GW2 = Array.zeroCreate (2 * H)
      GB2 = 0.0f
      GWwdl = Array.zeroCreate (3 * 2 * H)
      GBwdl = Array.zeroCreate 3
      LossCp = 0.0
      LossWdl = 0.0 }

let inline private addVec (dst: float32[]) (dstOff: int) (src: float32[]) (srcOff: int) =
    let vc = Vector<float32>.Count
    let mutable j = 0
    while j < H do
        let vd = Vector<float32>(dst, dstOff + j)
        let vs = Vector<float32>(src, srcOff + j)
        (vd + vs).CopyTo(dst, dstOff + j)
        j <- j + vc

let inline private sigmoidF (x: float32) = 1.0f / (1.0f + exp (-x))

/// Extract perspective features. Returns count, or -1 for a malformed record
/// (corrupt/misaligned data must be skipped, never crash the trainer).
let private extractFeatures (data: byte[]) (off: int) (fs: int[]) (fo: int[]) =
    let stm = int data.[off + 96]
    let wKing = BitConverter.ToUInt64(data, off + 5 * 8)
    let bKing = BitConverter.ToUInt64(data, off + 11 * 8)
    if stm > 1 || wKing = 0UL || bKing = 0UL then -1
    else
        let opp = stm ^^^ 1
        let wK = System.Numerics.BitOperations.TrailingZeroCount wKing
        let bK = System.Numerics.BitOperations.TrailingZeroCount bKing
        let kStm = if stm = 0 then wK else bK
        let kOpp = if stm = 0 then bK else wK
        let mutable cnt = 0
        let mutable ok = true
        let mutable pc = 0
        while ok && pc < 12 do
            let mutable bb = BitConverter.ToUInt64(data, off + pc * 8)
            while ok && bb <> 0UL do
                let sq = System.Numerics.BitOperations.TrailingZeroCount bb
                bb <- bb &&& (bb - 1UL)
                if cnt >= 64 then ok <- false
                else
                    fs.[cnt] <- AlterEgo.Nnue.featureIndexK stm pc sq kStm buckets mirror
                    fo.[cnt] <- AlterEgo.Nnue.featureIndexK opp pc sq kOpp buckets mirror
                    cnt <- cnt + 1
                    if dual then
                        fs.[cnt] <- AlterEgo.Nnue.featureIndexE stm pc sq kOpp
                        fo.[cnt] <- AlterEgo.Nnue.featureIndexE opp pc sq kStm
                        cnt <- cnt + 1
            pc <- pc + 1
        if ok then cnt else -1

/// One sample's forward + backward (CP + WDL heads), accumulating into g.
/// Also accumulates per-head losses into g. Returns the CP squared loss.
let private trainSample (net: Net) (g: Grad) (data: byte[]) (off: int)
                        (fs: int[]) (fo: int[]) (accS: float32[]) (accO: float32[]) =
    let cnt = extractFeatures data off fs fo
    if cnt < 0 then 0.0f else   // malformed record: contribute nothing
    let stm = int data.[off + 96]
    let scoreWhite = int (BitConverter.ToInt16(data, off + 97))
    let resultWhite = int (sbyte data.[off + 99])
    let scoreStm = if stm = 0 then scoreWhite else -scoreWhite
    let resStm = if stm = 0 then resultWhite else -resultWhite
    let target =
        0.7f * sigmoidF (float32 scoreStm / 200.0f)
        + 0.3f * (float32 (resStm + 1) * 0.5f)
    // WDL: pure game result, one-hot from stm POV (win/draw/loss)
    let wdlTargetIdx = if resStm > 0 then 0 elif resStm = 0 then 1 else 2

    // forward
    Array.blit net.B1 0 accS 0 H
    Array.blit net.B1 0 accO 0 H
    for i in 0 .. cnt - 1 do
        addVec accS 0 net.W1 (fs.[i] * H)
        addVec accO 0 net.W1 (fo.[i] * H)
    let mutable u = net.B2
    let mutable l0 = net.Bwdl.[0]
    let mutable l1 = net.Bwdl.[1]
    let mutable l2 = net.Bwdl.[2]
    for j in 0 .. H - 1 do
        let a = accS.[j]
        let c = if a < 0.0f then 0.0f elif a > 1.0f then 1.0f else a
        u <- u + c * net.W2.[j]
        l0 <- l0 + c * net.Wwdl.[j]
        l1 <- l1 + c * net.Wwdl.[2 * H + j]
        l2 <- l2 + c * net.Wwdl.[4 * H + j]
    for j in 0 .. H - 1 do
        let a = accO.[j]
        let c = if a < 0.0f then 0.0f elif a > 1.0f then 1.0f else a
        u <- u + c * net.W2.[H + j]
        l0 <- l0 + c * net.Wwdl.[H + j]
        l1 <- l1 + c * net.Wwdl.[2 * H + H + j]
        l2 <- l2 + c * net.Wwdl.[4 * H + H + j]

    let p = sigmoidF (2.0f * u)
    let err = p - target
    let dU = 4.0f * err * p * (1.0f - p)

    // WDL softmax + cross-entropy gradient (scaled by LambdaWdl)
    let m = max l0 (max l1 l2)
    let e0 = exp (l0 - m)
    let e1 = exp (l1 - m)
    let e2 = exp (l2 - m)
    let z = e0 + e1 + e2
    let p0 = e0 / z
    let p1 = e1 / z
    let p2 = e2 / z
    let t0 = if wdlTargetIdx = 0 then 1.0f else 0.0f
    let t1 = if wdlTargetIdx = 1 then 1.0f else 0.0f
    let t2 = if wdlTargetIdx = 2 then 1.0f else 0.0f
    let d0 = LambdaWdl * (p0 - t0)
    let d1 = LambdaWdl * (p1 - t1)
    let d2 = LambdaWdl * (p2 - t2)
    let pTrue = if wdlTargetIdx = 0 then p0 elif wdlTargetIdx = 1 then p1 else p2
    g.LossWdl <- g.LossWdl + float (-(log (max 1e-9f pTrue)))

    // backward
    g.GB2 <- g.GB2 + dU
    g.GBwdl.[0] <- g.GBwdl.[0] + d0
    g.GBwdl.[1] <- g.GBwdl.[1] + d1
    g.GBwdl.[2] <- g.GBwdl.[2] + d2
    for j in 0 .. H - 1 do
        let a = accS.[j]
        let c = if a < 0.0f then 0.0f elif a > 1.0f then 1.0f else a
        g.GW2.[j] <- g.GW2.[j] + dU * c
        g.GWwdl.[j] <- g.GWwdl.[j] + d0 * c
        g.GWwdl.[2 * H + j] <- g.GWwdl.[2 * H + j] + d1 * c
        g.GWwdl.[4 * H + j] <- g.GWwdl.[4 * H + j] + d2 * c
        // reuse accS as dAcc for the sparse W1 update below
        accS.[j] <-
            if a > 0.0f && a < 1.0f then
                dU * net.W2.[j] + d0 * net.Wwdl.[j] + d1 * net.Wwdl.[2 * H + j] + d2 * net.Wwdl.[4 * H + j]
            else 0.0f
    for j in 0 .. H - 1 do
        let a = accO.[j]
        let c = if a < 0.0f then 0.0f elif a > 1.0f then 1.0f else a
        g.GW2.[H + j] <- g.GW2.[H + j] + dU * c
        g.GWwdl.[H + j] <- g.GWwdl.[H + j] + d0 * c
        g.GWwdl.[2 * H + H + j] <- g.GWwdl.[2 * H + H + j] + d1 * c
        g.GWwdl.[4 * H + H + j] <- g.GWwdl.[4 * H + H + j] + d2 * c
        accO.[j] <-
            if a > 0.0f && a < 1.0f then
                dU * net.W2.[H + j] + d0 * net.Wwdl.[H + j] + d1 * net.Wwdl.[2 * H + H + j] + d2 * net.Wwdl.[4 * H + H + j]
            else 0.0f
    addVec g.GB1 0 accS 0
    addVec g.GB1 0 accO 0
    for i in 0 .. cnt - 1 do
        addVec g.GW1 (fs.[i] * H) accS 0
        addVec g.GW1 (fo.[i] * H) accO 0
    g.LossCp <- g.LossCp + float (err * err)
    err * err

// Adam optimizer state
type private Adam =
    { M: float32[]
      V: float32[] }

let private adamStep (w: float32[]) (a: Adam) (grad: float32[]) (lr: float32) (clip: float32) (scale: float32) =
    let b1 = 0.9f
    let b2 = 0.999f
    let eps = 1e-8f
    for i in 0 .. w.Length - 1 do
        let g = grad.[i] * scale
        a.M.[i] <- b1 * a.M.[i] + (1.0f - b1) * g
        a.V.[i] <- b2 * a.V.[i] + (1.0f - b2) * g * g
        let upd = lr * a.M.[i] / (sqrt a.V.[i] + eps)
        let nw = w.[i] - upd
        w.[i] <- if nw > clip then clip elif nw < -clip then -clip else nw

let private export (net: Net) (path: string) =
    let qa = float32 AlterEgo.Nnue.QA
    let qb = float32 AlterEgo.Nnue.QB
    let q (x: float32) (s: float32) =
        let v = int (round (float (x * s)))
        int16 (max -32767 (min 32767 v))
    let n: AlterEgo.Nnue.Network =
        { W1 = Array.map (fun w -> q w qa) net.W1
          B1 = Array.map (fun w -> q w qa) net.B1
          W2 = Array.map (fun w -> q w qb) net.W2
          B2 = int (round (float (net.B2 * qa * qb)))
          Buckets = buckets
          Mirror = mirror
          Dual = dual
          Wdl = Array.copy net.Wwdl     // float32 head: no quantization needed
          WdlBias = Array.copy net.Bwdl }
    AlterEgo.Nnue.save path n
    printfn "exported %s" path

/// Forward-only WDL reliability report over the holdout range: 10 bins of
/// predicted expected score (Pw + Pd/2) vs actual outcome.
let private calibrationReport (net: Net) (store: Store) (fromIdx: int) (toIdx: int) =
    let fs = Array.zeroCreate 66
    let fo = Array.zeroCreate 66
    let accS = Array.zeroCreate<float32> H
    let accO = Array.zeroCreate<float32> H
    let binPred = Array.zeroCreate<float> 10
    let binActual = Array.zeroCreate<float> 10
    let binCount = Array.zeroCreate<int> 10
    for k in fromIdx .. toIdx - 1 do
        let data = slabOf store k
        let off = offOf k
        let cnt = extractFeatures data off fs fo
        if cnt >= 0 then
            let stm = int data.[off + 96]
            let resultWhite = int (sbyte data.[off + 99])
            let resStm = if stm = 0 then resultWhite else -resultWhite
            Array.blit net.B1 0 accS 0 H
            Array.blit net.B1 0 accO 0 H
            for i in 0 .. cnt - 1 do
                addVec accS 0 net.W1 (fs.[i] * H)
                addVec accO 0 net.W1 (fo.[i] * H)
            let mutable l0 = net.Bwdl.[0]
            let mutable l1 = net.Bwdl.[1]
            let mutable l2 = net.Bwdl.[2]
            for j in 0 .. H - 1 do
                let cS = (let a = accS.[j] in if a < 0.0f then 0.0f elif a > 1.0f then 1.0f else a)
                let cO = (let a = accO.[j] in if a < 0.0f then 0.0f elif a > 1.0f then 1.0f else a)
                l0 <- l0 + cS * net.Wwdl.[j] + cO * net.Wwdl.[H + j]
                l1 <- l1 + cS * net.Wwdl.[2 * H + j] + cO * net.Wwdl.[2 * H + H + j]
                l2 <- l2 + cS * net.Wwdl.[4 * H + j] + cO * net.Wwdl.[4 * H + H + j]
            let m = max l0 (max l1 l2)
            let e0 = exp (l0 - m)
            let e1 = exp (l1 - m)
            let e2 = exp (l2 - m)
            let z = e0 + e1 + e2
            let predScore = float ((e0 + 0.5f * e1) / z)
            let actual = float (resStm + 1) * 0.5
            let bin = min 9 (int (predScore * 10.0))
            binPred.[bin] <- binPred.[bin] + predScore
            binActual.[bin] <- binActual.[bin] + actual
            binCount.[bin] <- binCount.[bin] + 1
    printfn "WDL calibration (holdout): bin  predicted  actual  n"
    let mutable ece = 0.0
    let mutable total = 0
    for b in 0 .. 9 do
        if binCount.[b] > 0 then
            let p = binPred.[b] / float binCount.[b]
            let a = binActual.[b] / float binCount.[b]
            ece <- ece + abs (p - a) * float binCount.[b]
            total <- total + binCount.[b]
            printfn "  %d0%%  %.3f  %.3f  %d" b p a binCount.[b]
    if total > 0 then
        printfn "  expected calibration error: %.4f" (ece / float total)

let run (dataPath: string) (epochs: int) (outPath: string) (kingBuckets: int) =
    // 1 = flat 768; 4 = own-king buckets (3072); 8 = factorized dual-king (6144)
    dual <- kingBuckets >= 8
    buckets <- if dual then 4 else max 1 (min 4 kingBuckets)
    mirror <- buckets > 1
    In <- if dual then 6144 else 768 * buckets
    printfn "architecture: (%d -> %d)x2 -> 1  (%d king buckets%s%s)"
        In H buckets (if mirror then ", mirrored" else "") (if dual then ", dual-king factorized" else "")
    let store = loadStore dataPath
    let total = store.Total
    let valCount = max 1 (total / 100)
    let trainCount = total - valCount
    printfn "training on %d samples (%d held out), %d epochs" trainCount valCount epochs

    let rng = Random 42
    let net = newNet rng
    let adamW1 = { M = Array.zeroCreate (In * H); V = Array.zeroCreate (In * H) }
    let adamB1 = { M = Array.zeroCreate H; V = Array.zeroCreate H }
    let adamW2 = { M = Array.zeroCreate (2 * H); V = Array.zeroCreate (2 * H) }
    let adamB2 = { M = Array.zeroCreate 1; V = Array.zeroCreate 1 }
    let adamWwdl = { M = Array.zeroCreate (3 * 2 * H); V = Array.zeroCreate (3 * 2 * H) }
    let adamBwdl = { M = Array.zeroCreate 3; V = Array.zeroCreate 3 }

    let threads = max 1 (Environment.ProcessorCount - 1)
    let grads = Array.init threads (fun _ -> newGrad ())
    // tiny datasets: one giant batch would mean a single Adam step per epoch;
    // keep >=8 steps per epoch (no-op for every dataset >= 128K samples)
    let batchSize = min 16384 (max 256 ((trainCount + 7) / 8))
    let perm = Array.init trainCount id
    let mutable lr = 0.0015f

    let sw = System.Diagnostics.Stopwatch.StartNew()
    let mutable lastTick = 0.0   // intra-epoch progress, at most every 30s
    for epoch in 1 .. epochs do
        // shuffle
        for i in trainCount - 1 .. -1 .. 1 do
            let j = rng.Next(i + 1)
            let t = perm.[i] in perm.[i] <- perm.[j]; perm.[j] <- t
        let mutable epochLoss = 0.0
        let mutable epochWdl = 0.0
        let mutable batches = 0
        let mutable b = 0
        while b < trainCount do
            let count = min batchSize (trainCount - b)
            let chunk = (count + threads - 1) / threads
            let losses = Array.zeroCreate<float> threads
            Parallel.For(0, threads, fun t ->
                let g = grads.[t]
                Array.Clear(g.GW1, 0, g.GW1.Length)
                Array.Clear(g.GB1, 0, g.GB1.Length)
                Array.Clear(g.GW2, 0, g.GW2.Length)
                Array.Clear(g.GWwdl, 0, g.GWwdl.Length)
                Array.Clear(g.GBwdl, 0, g.GBwdl.Length)
                g.GB2 <- 0.0f
                g.LossCp <- 0.0
                g.LossWdl <- 0.0
                let fs = Array.zeroCreate 66
                let fo = Array.zeroCreate 66
                let accS = Array.zeroCreate<float32> H
                let accO = Array.zeroCreate<float32> H
                let lo = b + t * chunk
                let hi = min (b + count) (lo + chunk)
                let mutable loss = 0.0
                for k in lo .. hi - 1 do
                    let idx = perm.[k]
                    loss <- loss + float (trainSample net g (slabOf store idx) (offOf idx) fs fo accS accO)
                losses.[t] <- loss) |> ignore
            // reduce thread gradients into grads.[0]
            let g0 = grads.[0]
            for t in 1 .. threads - 1 do
                let gt = grads.[t]
                for i in 0 .. g0.GW1.Length - 1 do g0.GW1.[i] <- g0.GW1.[i] + gt.GW1.[i]
                for i in 0 .. g0.GB1.Length - 1 do g0.GB1.[i] <- g0.GB1.[i] + gt.GB1.[i]
                for i in 0 .. g0.GW2.Length - 1 do g0.GW2.[i] <- g0.GW2.[i] + gt.GW2.[i]
                for i in 0 .. g0.GWwdl.Length - 1 do g0.GWwdl.[i] <- g0.GWwdl.[i] + gt.GWwdl.[i]
                for i in 0 .. 2 do g0.GBwdl.[i] <- g0.GBwdl.[i] + gt.GBwdl.[i]
                g0.GB2 <- g0.GB2 + gt.GB2
                g0.LossWdl <- g0.LossWdl + gt.LossWdl
            let scale = 1.0f / float32 count
            adamStep net.W1 adamW1 g0.GW1 lr 1.98f scale
            adamStep net.B1 adamB1 g0.GB1 lr 1.98f scale
            adamStep net.W2 adamW2 g0.GW2 lr 1.98f scale
            adamStep net.Wwdl adamWwdl g0.GWwdl lr 10.0f scale   // float head: wide clip
            adamStep net.Bwdl adamBwdl g0.GBwdl lr 10.0f scale
            // B2 scalar via tiny arrays
            let gb2 = [| g0.GB2 |]
            let wb2 = [| net.B2 |]
            adamStep wb2 adamB2 gb2 lr 1000.0f scale
            net.B2 <- wb2.[0]
            epochLoss <- epochLoss + Array.sum losses
            epochWdl <- epochWdl + g0.LossWdl
            batches <- batches + 1
            b <- b + count
            if sw.Elapsed.TotalSeconds - lastTick >= 30.0 then
                lastTick <- sw.Elapsed.TotalSeconds
                let frac = (float (epoch - 1) + float b / float trainCount) / float epochs
                printfn "  %.1f%%  elapsed %.0fs  ETA %.0fs"
                    (frac * 100.0) sw.Elapsed.TotalSeconds (sw.Elapsed.TotalSeconds * (1.0 - frac) / frac)
        // validation
        let fs = Array.zeroCreate 66
        let fo = Array.zeroCreate 66
        let accS = Array.zeroCreate<float32> H
        let accO = Array.zeroCreate<float32> H
        let gDummy = newGrad ()
        let mutable valLoss = 0.0
        for k in trainCount .. total - 1 do
            valLoss <- valLoss + float (trainSample net gDummy (slabOf store k) (offOf k) fs fo accS accO)
        printfn "epoch %2d/%d  cp %.6f/%.6f  wdl %.4f/%.4f  lr %.5f  (%.0fs  %d%%  ETA %.0fs)"
            epoch epochs (epochLoss / float trainCount) (valLoss / float valCount)
            (epochWdl / float trainCount) (gDummy.LossWdl / float valCount)
            lr sw.Elapsed.TotalSeconds (epoch * 100 / epochs)
            (sw.Elapsed.TotalSeconds / float epoch * float (epochs - epoch))
        if epoch % 6 = 0 then lr <- lr * 0.5f
        export net outPath
    calibrationReport net store trainCount total
    printfn "done in %.0fs" sw.Elapsed.TotalSeconds
