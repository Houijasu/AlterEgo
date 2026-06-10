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

// input size is runtime: 768 * kingBuckets (set once by run)
let mutable private In = 768
let mutable private buckets = 1
let mutable private mirror = false

type private Net =
    { W1: float32[]      // In*H, feature-major
      B1: float32[]
      W2: float32[]      // 2H
      mutable B2: float32 }

let private newNet (rng: Random) =
    let scale1 = sqrt (2.0 / 32.0) |> float32      // ~32 active features
    let scale2 = sqrt (2.0 / float (2 * H)) |> float32
    { W1 = Array.init (In * H) (fun _ -> (float32 (rng.NextDouble()) - 0.5f) * 2.0f * scale1)
      B1 = Array.zeroCreate H
      W2 = Array.init (2 * H) (fun _ -> (float32 (rng.NextDouble()) - 0.5f) * 2.0f * scale2)
      B2 = 0.0f }

type private Grad =
    { GW1: float32[]
      GB1: float32[]
      GW2: float32[]
      mutable GB2: float32 }

let private newGrad () =
    { GW1 = Array.zeroCreate (In * H)
      GB1 = Array.zeroCreate H
      GW2 = Array.zeroCreate (2 * H)
      GB2 = 0.0f }

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
                if cnt >= 32 then ok <- false
                else
                    fs.[cnt] <- AlterEgo.Nnue.featureIndexK stm pc sq kStm buckets mirror
                    fo.[cnt] <- AlterEgo.Nnue.featureIndexK opp pc sq kOpp buckets mirror
                    cnt <- cnt + 1
            pc <- pc + 1
        if ok then cnt else -1

/// One sample's forward + backward, accumulating into g. Returns squared loss.
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

    // forward
    Array.blit net.B1 0 accS 0 H
    Array.blit net.B1 0 accO 0 H
    for i in 0 .. cnt - 1 do
        addVec accS 0 net.W1 (fs.[i] * H)
        addVec accO 0 net.W1 (fo.[i] * H)
    let mutable u = net.B2
    for j in 0 .. H - 1 do
        let a = accS.[j]
        let c = if a < 0.0f then 0.0f elif a > 1.0f then 1.0f else a
        u <- u + c * net.W2.[j]
    for j in 0 .. H - 1 do
        let a = accO.[j]
        let c = if a < 0.0f then 0.0f elif a > 1.0f then 1.0f else a
        u <- u + c * net.W2.[H + j]

    let p = sigmoidF (2.0f * u)
    let err = p - target
    let dU = 4.0f * err * p * (1.0f - p)

    // backward
    g.GB2 <- g.GB2 + dU
    for j in 0 .. H - 1 do
        let a = accS.[j]
        let c = if a < 0.0f then 0.0f elif a > 1.0f then 1.0f else a
        g.GW2.[j] <- g.GW2.[j] + dU * c
        // reuse accS as dAcc for the sparse W1 update below
        accS.[j] <- if a > 0.0f && a < 1.0f then dU * net.W2.[j] else 0.0f
    for j in 0 .. H - 1 do
        let a = accO.[j]
        let c = if a < 0.0f then 0.0f elif a > 1.0f then 1.0f else a
        g.GW2.[H + j] <- g.GW2.[H + j] + dU * c
        accO.[j] <- if a > 0.0f && a < 1.0f then dU * net.W2.[H + j] else 0.0f
    addVec g.GB1 0 accS 0
    addVec g.GB1 0 accO 0
    for i in 0 .. cnt - 1 do
        addVec g.GW1 (fs.[i] * H) accS 0
        addVec g.GW1 (fo.[i] * H) accO 0
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
          Mirror = mirror }
    AlterEgo.Nnue.save path n
    printfn "exported %s" path

let run (dataPath: string) (epochs: int) (outPath: string) (kingBuckets: int) =
    buckets <- max 1 (min 4 kingBuckets)
    mirror <- buckets > 1
    In <- 768 * buckets
    printfn "architecture: (%d -> %d)x2 -> 1  (%d king buckets%s)"
        In H buckets (if mirror then ", mirrored" else "")
    let data = File.ReadAllBytes dataPath
    let total = data.Length / SampleBytes
    let valCount = max 1 (total / 100)
    let trainCount = total - valCount
    printfn "training on %d samples (%d held out), %d epochs" trainCount valCount epochs

    let rng = Random 42
    let net = newNet rng
    let adamW1 = { M = Array.zeroCreate (In * H); V = Array.zeroCreate (In * H) }
    let adamB1 = { M = Array.zeroCreate H; V = Array.zeroCreate H }
    let adamW2 = { M = Array.zeroCreate (2 * H); V = Array.zeroCreate (2 * H) }
    let adamB2 = { M = Array.zeroCreate 1; V = Array.zeroCreate 1 }

    let threads = max 1 (Environment.ProcessorCount - 1)
    let grads = Array.init threads (fun _ -> newGrad ())
    let batchSize = 16384
    let perm = Array.init trainCount id
    let mutable lr = 0.0015f

    let sw = System.Diagnostics.Stopwatch.StartNew()
    for epoch in 1 .. epochs do
        // shuffle
        for i in trainCount - 1 .. -1 .. 1 do
            let j = rng.Next(i + 1)
            let t = perm.[i] in perm.[i] <- perm.[j]; perm.[j] <- t
        let mutable epochLoss = 0.0
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
                g.GB2 <- 0.0f
                let fs = Array.zeroCreate 33
                let fo = Array.zeroCreate 33
                let accS = Array.zeroCreate<float32> H
                let accO = Array.zeroCreate<float32> H
                let lo = b + t * chunk
                let hi = min (b + count) (lo + chunk)
                let mutable loss = 0.0
                for k in lo .. hi - 1 do
                    loss <- loss + float (trainSample net g data (perm.[k] * SampleBytes) fs fo accS accO)
                losses.[t] <- loss) |> ignore
            // reduce thread gradients into grads.[0]
            let g0 = grads.[0]
            for t in 1 .. threads - 1 do
                let gt = grads.[t]
                for i in 0 .. g0.GW1.Length - 1 do g0.GW1.[i] <- g0.GW1.[i] + gt.GW1.[i]
                for i in 0 .. g0.GB1.Length - 1 do g0.GB1.[i] <- g0.GB1.[i] + gt.GB1.[i]
                for i in 0 .. g0.GW2.Length - 1 do g0.GW2.[i] <- g0.GW2.[i] + gt.GW2.[i]
                g0.GB2 <- g0.GB2 + gt.GB2
            let scale = 1.0f / float32 count
            adamStep net.W1 adamW1 g0.GW1 lr 1.98f scale
            adamStep net.B1 adamB1 g0.GB1 lr 1.98f scale
            adamStep net.W2 adamW2 g0.GW2 lr 1.98f scale
            // B2 scalar via tiny arrays
            let gb2 = [| g0.GB2 |]
            let wb2 = [| net.B2 |]
            adamStep wb2 adamB2 gb2 lr 1000.0f scale
            net.B2 <- wb2.[0]
            epochLoss <- epochLoss + Array.sum losses
            batches <- batches + 1
            b <- b + count
        // validation
        let fs = Array.zeroCreate 33
        let fo = Array.zeroCreate 33
        let accS = Array.zeroCreate<float32> H
        let accO = Array.zeroCreate<float32> H
        let gDummy = newGrad ()
        let mutable valLoss = 0.0
        for k in trainCount .. total - 1 do
            valLoss <- valLoss + float (trainSample net gDummy data (k * SampleBytes) fs fo accS accO)
        printfn "epoch %2d  train %.6f  val %.6f  lr %.5f  (%.0fs)"
            epoch (epochLoss / float trainCount) (valLoss / float valCount) lr sw.Elapsed.TotalSeconds
        if epoch % 6 = 0 then lr <- lr * 0.5f
        export net outPath
    printfn "done in %.0fs" sw.Elapsed.TotalSeconds
