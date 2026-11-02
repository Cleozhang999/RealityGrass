import Testing
import simd
@testable import RealityGrass

@Suite struct GrassScatterTests {
    private let flat = GroundSample.grid(size: [2, 2], resolution: [11, 11]) { _, _ in 0 }

    @Test func densityMatchesArea() {
        let roots = GrassScatter(density: 100).roots(on: flat)
        #expect(roots.count == 400) // 2 m × 2 m × 100 per m²
    }

    @Test func isDeterministicForASeed() {
        let a = GrassScatter(density: 50, seed: 7).roots(on: flat)
        let b = GrassScatter(density: 50, seed: 7).roots(on: flat)
        let c = GrassScatter(density: 50, seed: 8).roots(on: flat)
        #expect(a == b)
        #expect(a != c)
    }

    @Test func respectsMaxBlades() {
        #expect(GrassScatter(density: 10_000, maxBlades: 123).roots(on: flat).count == 123)
    }

    @Test func skipsGroundBelowMinHeight() {
        // Left half is underwater at y = -1.
        let ground = GroundSample.grid(size: [2, 2], resolution: [21, 21]) { x, _ in x < 0 ? -1 : 0 }
        let roots = GrassScatter(density: 100, minHeight: -0.5, jitter: 0).roots(on: ground)
        #expect(!roots.isEmpty)
        #expect(roots.allSatisfy { $0.y >= -0.5 })
    }

    @Test func skipsSteepGround() {
        let wall = [GroundSample(position: [0, 0, 0], normal: [1, 0, 0]),
                    GroundSample(position: [1, 0, 1], normal: [1, 0, 0])]
        #expect(GrassScatter(density: 100).roots(on: wall).isEmpty)
    }

    @Test func gridNormalsFollowSlope() {
        // y = x rises at 45°, so the normal tilts towards -X.
        let ramp = GroundSample.grid(size: [2, 2], resolution: [3, 3]) { x, _ in x }
        let normal = ramp[4].normal
        #expect(abs(normal.x + normal.y) < 1e-5)
        #expect(abs(normal.y - 1 / sqrt(2)) < 1e-5)
    }
}

@Suite struct GrassLODTests {
    @Test func picksLevelByDistance() {
        let lod = GrassLOD.default
        #expect(lod.updateInterval(atDistance: 1) == 1)
        #expect(lod.updateInterval(atDistance: 7) == 2)
        #expect(lod.updateInterval(atDistance: 500) == 6)
    }

    @Test func disabledAlwaysUpdates() {
        #expect(GrassLOD.disabled.updateInterval(atDistance: 1000) == 1)
    }
}

@Suite struct GrassWindTests {
    @Test func directionIsFlattened() {
        #expect(GrassWind(direction: [0, 5, 2]).horizontalDirection == [0, 0, 1])
        #expect(GrassWind(direction: [0, 1, 0]).horizontalDirection == [1, 0, 0])
    }
}

@Suite struct GrassPaletteTests {
    @Test func rootIsDarkerThanTip() {
        let palette = GrassPalette.meadow
        for tone in stride(from: Float(0), to: 1, by: 0.1) {
            let root = palette.color(tone: tone, along: 0)
            let tip = palette.color(tone: tone, along: 1)
            #expect(simd_reduce_add(root) < simd_reduce_add(tip) * 0.5)
        }
    }

    @Test func dryShareUsesDryColor() {
        let palette = GrassPalette(blades: [[0, 1, 0]], dryShare: 0.25, dry: [1, 0, 0], rootBrightness: 1, tipBrightness: 1)
        #expect(palette.color(tone: 0.1, along: 0) == [0, 1, 0])
        #expect(palette.color(tone: 0.9, along: 0) == [1, 0, 0])
    }

    @Test func textureIsSquare() {
        let image = GrassMaterial.makeImage(palette: .meadow)
        #expect(image.width == 64 && image.height == 64)
    }
}
