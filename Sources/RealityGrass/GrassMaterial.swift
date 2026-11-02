import CoreGraphics
import Foundation
import RealityKit

/// Colors for the procedural grass texture. Components are sRGB in [0, 1].
public struct GrassPalette: Sendable, Equatable {
    /// The colors blades are tinted with. Each blade picks a random spot along this list.
    public var blades: [SIMD3<Float>]
    /// Share of blades, from 0 to 1, that are dry and use `dry` instead.
    public var dryShare: Float
    public var dry: SIMD3<Float>
    /// How dark the root is compared with the middle of the blade. 0 is black, 1 is no darkening.
    public var rootBrightness: Float
    /// Extra brightness at the tip, where the blade catches the most light.
    public var tipBrightness: Float

    public init(
        blades: [SIMD3<Float>],
        dryShare: Float = 0.08,
        dry: SIMD3<Float> = [0.66, 0.60, 0.33],
        rootBrightness: Float = 0.3,
        tipBrightness: Float = 1.2
    ) {
        precondition(!blades.isEmpty, "A palette needs at least one blade color")
        self.blades = blades
        self.dryShare = dryShare
        self.dry = dry
        self.rootBrightness = rootBrightness
        self.tipBrightness = tipBrightness
    }

    /// Fresh spring greens.
    public static let meadow = GrassPalette(blades: [
        [0.36, 0.58, 0.17], [0.27, 0.50, 0.14], [0.42, 0.62, 0.22], [0.22, 0.42, 0.13], [0.46, 0.60, 0.20],
    ])

    /// Desaturated sage and olive greens, for a calmer, more natural look such as marshland.
    public static let muted = GrassPalette(blades: [
        [0.42, 0.49, 0.32], [0.35, 0.42, 0.28], [0.47, 0.51, 0.36], [0.31, 0.37, 0.25], [0.44, 0.47, 0.33],
    ], dryShare: 0.14, dry: [0.60, 0.56, 0.42], rootBrightness: 0.32, tipBrightness: 1.15)

    /// Late-summer grass, more olive with plenty of dry blades.
    public static let lateSummer = GrassPalette(blades: [
        [0.45, 0.52, 0.20], [0.38, 0.46, 0.17], [0.52, 0.55, 0.25],
    ], dryShare: 0.3)

    /// The color of a blade with random value `tone` at `along` (0 = root, 1 = tip).
    func color(tone: Float, along: Float) -> SIMD3<Float> {
        var base: SIMD3<Float>
        if tone >= 1 - dryShare {
            base = dry
        } else {
            // Blend smoothly between neighboring palette entries.
            let position = tone / max(1 - dryShare, 1e-5) * Float(blades.count - 1)
            let index = min(Int(position), blades.count - 1)
            let next = min(index + 1, blades.count - 1)
            base = simd_mix(blades[index], blades[next], SIMD3(repeating: position - Float(index)))
        }
        // Dark at the root (self-shadowing in a dense lawn), brightest near the tip.
        let shade = rootBrightness + (tipBrightness - rootBrightness) * pow(along, 0.6)
        let sunlit = simd_mix(base, SIMD3(0.85, 0.85, 0.45), SIMD3(repeating: 0.12 * along * along))
        return simd_clamp(sunlit * shade, SIMD3(repeating: 0), SIMD3(repeating: 1))
    }
}

/// Builds materials for `GrassFieldEntity`.
public enum GrassMaterial {
    /// A double-sided material with a procedural texture: each blade gets a color from the
    /// palette, with a gradient from a dark root to a bright tip.
    @MainActor
    public static func make(palette: GrassPalette = .meadow, roughness: Float = 0.75) async throws -> PhysicallyBasedMaterial {
        let texture = try await TextureResource(image: makeImage(palette: palette), options: .init(semantic: .color))
        var material = PhysicallyBasedMaterial()
        material.baseColor = .init(texture: .init(texture))
        material.roughness = .init(floatLiteral: roughness)
        material.metallic = 0.0
        material.faceCulling = .none
        return material
    }

    /// The texture image. Horizontal axis: per-blade tone (UV `u`). Vertical axis: root (bottom) to tip (top).
    static func makeImage(palette: GrassPalette, size: Int = 64) -> CGImage {
        var pixels = [UInt8](repeating: 255, count: size * size * 4)
        for row in 0..<size {
            // Image rows run top to bottom, while UV `v` runs bottom to top.
            let along = 1 - Float(row) / Float(size - 1)
            for column in 0..<size {
                let color = palette.color(tone: (Float(column) + 0.5) / Float(size), along: along)
                let i = (row * size + column) * 4
                pixels[i] = UInt8(color.x * 255)
                pixels[i + 1] = UInt8(color.y * 255)
                pixels[i + 2] = UInt8(color.z * 255)
            }
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: size * 4,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)!
    }
}
