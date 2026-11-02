import simd

/// A point on the ground, with the surface normal at that point. Y is up.
public struct GroundSample: Sendable, Equatable {
    public var position: SIMD3<Float>
    public var normal: SIMD3<Float>

    public init(position: SIMD3<Float>, normal: SIMD3<Float> = [0, 1, 0]) {
        self.position = position
        self.normal = normal
    }

    /// Samples a height field on a regular grid centered on the origin.
    ///
    /// - Parameters:
    ///   - size: Width (X) and depth (Z) of the area in meters.
    ///   - resolution: Number of samples along X and Z. Each must be at least 2.
    ///   - height: Returns the ground height (Y) at an (x, z) position.
    public static func grid(
        size: SIMD2<Float>,
        resolution: SIMD2<Int>,
        height: (_ x: Float, _ z: Float) -> Float
    ) -> [GroundSample] {
        precondition(resolution.x >= 2 && resolution.y >= 2, "resolution must be at least 2 in each dimension")
        let step = size / SIMD2<Float>(Float(resolution.x - 1), Float(resolution.y - 1))
        var samples: [GroundSample] = []
        samples.reserveCapacity(resolution.x * resolution.y)
        for row in 0..<resolution.y {
            for column in 0..<resolution.x {
                let x = Float(column) * step.x - size.x / 2
                let z = Float(row) * step.y - size.y / 2
                // Central differences for the normal.
                let dx = height(x + step.x / 2, z) - height(x - step.x / 2, z)
                let dz = height(x, z + step.y / 2) - height(x, z - step.y / 2)
                let normal = simd_normalize(SIMD3<Float>(-dx / step.x, 1, -dz / step.y))
                samples.append(GroundSample(position: [x, height(x, z), z], normal: normal))
            }
        }
        return samples
    }
}

/// Chooses where grass blades grow on a set of ground samples.
public struct GrassScatter: Sendable, Equatable {
    /// Blades per square meter of the samples' horizontal bounding box.
    public var density: Float
    /// Steepest ground, in radians from horizontal, that grass grows on.
    public var maxSlope: Float
    /// Grass only grows at or above this height. Use it to keep grass out of water.
    public var minHeight: Float
    /// Maximum horizontal offset, in meters, from the sample a blade is placed at.
    /// Use roughly half the sample spacing to hide the grid.
    public var jitter: Float
    /// Upper bound on the number of blades, to keep GPU memory in check.
    public var maxBlades: Int
    /// Seed for the random placement, so the same inputs always produce the same field.
    public var seed: UInt64

    public init(
        density: Float = 400,
        maxSlope: Float = .pi / 4,
        minHeight: Float = -.infinity,
        jitter: Float = 0.05,
        maxBlades: Int = 200_000,
        seed: UInt64 = 0
    ) {
        self.density = density
        self.maxSlope = maxSlope
        self.minHeight = minHeight
        self.jitter = jitter
        self.maxBlades = maxBlades
        self.seed = seed
    }

    /// Returns the root position of each blade.
    public func roots(on samples: [GroundSample]) -> [SIMD3<Float>] {
        let minUpComponent = cos(maxSlope)
        let fertile = samples.filter { sample in
            sample.position.y >= minHeight && simd_normalize(sample.normal).y >= minUpComponent
        }
        guard !fertile.isEmpty, density > 0 else { return [] }

        var lower = SIMD2<Float>(repeating: .infinity)
        var upper = SIMD2<Float>(repeating: -.infinity)
        for sample in samples {
            let xz = SIMD2(sample.position.x, sample.position.z)
            lower = simd_min(lower, xz)
            upper = simd_max(upper, xz)
        }
        let extent = upper - lower
        let area = max(extent.x * extent.y, 0)
        // Only the fertile share of the area gets grass.
        let fertileArea = area * Float(fertile.count) / Float(samples.count)
        let count = min(Int((fertileArea * density).rounded()), maxBlades)

        var random = SplitMix64(seed: seed)
        var roots: [SIMD3<Float>] = []
        roots.reserveCapacity(count)
        for _ in 0..<count {
            let sample = fertile[Int(random.next() % UInt64(fertile.count))]
            let offset = SIMD3<Float>(
                Float.random(in: -jitter...jitter, using: &random),
                0,
                Float.random(in: -jitter...jitter, using: &random)
            )
            roots.append(sample.position + offset)
        }
        return roots
    }
}

/// A small, fast, seedable generator, so scattering is reproducible across runs.
struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
