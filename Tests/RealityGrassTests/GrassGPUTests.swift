import GrassShaderTypes
import Metal
import RealityKit
import Testing
@testable import RealityGrass

/// Runs the compute shaders on the GPU and checks the mesh they produce.
@MainActor
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Needs a Metal device"))
struct GrassGPUTests {
    private func vertices(of entity: GrassFieldEntity) -> [GrassVertex] {
        let mesh = entity.components[GrassComponent.self]!.state.mesh
        var result: [GrassVertex] = []
        mesh.withUnsafeBytes(bufferIndex: 0) { result = Array($0.bindMemory(to: GrassVertex.self)) }
        return result
    }

    private func indices(of entity: GrassFieldEntity) -> [UInt32] {
        let mesh = entity.components[GrassComponent.self]!.state.mesh
        var result: [UInt32] = []
        mesh.withUnsafeIndices { result = Array($0.bindMemory(to: UInt32.self)) }
        return result
    }

    @Test func generatesUprightBladesAtTheirRoots() async throws {
        let roots: [SIMD3<Float>] = [[0, 0, 0], [1, 0.5, -1], [-2, 0, 3]]
        let shape = GrassBladeShape(height: 0.3, width: 0.02, curvature: 0.15)
        let grass = try await GrassFieldEntity(roots: roots, shape: shape, wind: .calm)
        #expect(grass.bladeCount == 3)

        let vertices = vertices(of: grass)
        #expect(vertices.count == 3 * Int(GRASS_VERTICES_PER_BLADE))
        for (blade, root) in roots.enumerated() {
            let base = blade * Int(GRASS_VERTICES_PER_BLADE)
            let rootMid = (vertices[base].position + vertices[base + 1].position) / 2
            #expect(simd_distance(rootMid, root) < 1e-4, "blade \(blade) root")

            let tip = (vertices[base + 6].position + vertices[base + 7].position) / 2
            let height = tip.y - root.y
            // Blades range from about 0.45 to 1.75 times the nominal height, minus lean and curl.
            #expect(height > 0.3 * 0.3 && height <= 0.3 * 1.8, "blade \(blade) height \(height)")
            #expect(simd_distance(tip, root) <= 0.3 * 1.8, "blade \(blade) too long")

            // v runs from root to tip; u is one random tone for the whole blade.
            #expect(vertices[base].uv.y == 0 && abs(vertices[base + 7].uv.y - 1) < 1e-5)
            let tone = vertices[base].uv.x
            #expect(tone >= 0 && tone < 1)
            #expect((base..<base + 8).allSatisfy { vertices[$0].uv.x == tone })
            for i in base..<base + 8 {
                #expect(abs(simd_length(vertices[i].normal) - 1) < 1e-3)
            }
        }
    }

    @Test func writesValidIndices() async throws {
        let grass = try await GrassFieldEntity(roots: [[0, 0, 0], [1, 0, 0]], wind: .calm)
        let indices = indices(of: grass)
        #expect(indices.count == 2 * Int(GRASS_INDICES_PER_BLADE))
        #expect(indices.allSatisfy { $0 < 2 * GRASS_VERTICES_PER_BLADE })
        // Second blade only references its own vertices.
        #expect(indices[Int(GRASS_INDICES_PER_BLADE)...].allSatisfy { $0 >= GRASS_VERTICES_PER_BLADE })
    }

    @Test func windBendsTipsDownwind() async throws {
        let roots = (0..<64).map { SIMD3<Float>(Float($0), 0, 0) }
        let calm = try await GrassFieldEntity(roots: roots, wind: .calm)
        let windy = try await GrassFieldEntity(roots: roots, wind: .calm)

        // Pose both at a moment where sin(time * frequency + phase) is non-zero for most blades,
        // using the same encoding path as the per-frame system.
        let gpu = try GrassGPU.shared()
        let wind = GrassWind(strength: 0.8, frequency: 1, direction: [0, 0, 1])
        let state = windy.components[GrassComponent.self]!.state
        var params = state.params(wind: wind, time: 1.3)
        let commandBuffer = try #require(gpu.queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        state.encodeAnimation(into: encoder, commandBuffer: commandBuffer, params: &params, gpu: gpu)
        encoder.endEncoding()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            commandBuffer.addCompletedHandler { _ in continuation.resume() }
            commandBuffer.commit()
        }

        let rest = vertices(of: calm)
        let bent = vertices(of: windy)
        var moved = 0
        for blade in roots.indices {
            let tip = blade * 8 + 6
            let root = blade * 8
            #expect(simd_distance(bent[root].position, rest[root].position) < 1e-5, "roots must not move")
            let dz = bent[tip].position.z - rest[tip].position.z
            let dx = bent[tip].position.x - rest[tip].position.x
            // Bending is mainly downwind (Z). Gusts may veer a little to the side.
            #expect(abs(dx) <= abs(dz) * 0.6 + 1e-3)
            if abs(dz) > 1e-3 { moved += 1 }
            // Bending rotates around the root midpoint, so distance from it is preserved.
            let pivot = (rest[root].position + rest[root + 1].position) / 2
            let restLength = simd_distance(rest[tip].position, pivot)
            let bentLength = simd_distance(bent[tip].position, pivot)
            #expect(abs(restLength - bentLength) < 1e-4)
        }
        #expect(moved > roots.count / 2)
    }

    /// Bends a grid of blades with `wind` at `time` and returns each blade's bend angle in radians, row by row.
    private func tipDisplacements(side: Int, spacing: Float, wind: GrassWind, time: Float) async throws -> [[Float]] {
        var roots: [SIMD3<Float>] = []
        for row in 0..<side {
            for column in 0..<side {
                roots.append([Float(column) * spacing, 0, Float(row) * spacing])
            }
        }
        let shape = GrassBladeShape(height: 0.3, width: 0.02, curvature: 0)
        let calm = try await GrassFieldEntity(roots: roots, shape: shape, wind: .calm)
        let windy = try await GrassFieldEntity(roots: roots, shape: shape, wind: .calm)

        let gpu = try GrassGPU.shared()
        let state = windy.components[GrassComponent.self]!.state
        var params = state.params(wind: wind, time: time)
        let commandBuffer = try #require(gpu.queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        state.encodeAnimation(into: encoder, commandBuffer: commandBuffer, params: &params, gpu: gpu)
        encoder.endEncoding()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            commandBuffer.addCompletedHandler { _ in continuation.resume() }
            commandBuffer.commit()
        }

        let rest = vertices(of: calm), bent = vertices(of: windy)
        return (0..<side).map { row in
            (0..<side).map { column in
                let base = (row * side + column) * 8
                let root = (rest[base].position + rest[base + 1].position) / 2
                let restTip = (rest[base + 6].position + rest[base + 7].position) / 2 - root
                let bentTip = (bent[base + 6].position + bent[base + 7].position) / 2 - root
                return acos(min(simd_dot(simd_normalize(restTip), simd_normalize(bentTip)), 1))
            }
        }
    }

    private func correlation(_ a: [Float], _ b: [Float]) -> Float {
        let meanA = a.reduce(0, +) / Float(a.count), meanB = b.reduce(0, +) / Float(b.count)
        var covariance: Float = 0, varianceA: Float = 0, varianceB: Float = 0
        for (x, y) in zip(a, b) {
            covariance += (x - meanA) * (y - meanB)
            varianceA += (x - meanA) * (x - meanA)
            varianceB += (y - meanB) * (y - meanB)
        }
        return covariance / (varianceA * varianceB).squareRoot()
    }

    @Test func gustsArriveInTravellingClumps() async throws {
        let wind = GrassWind(strength: 0.8, frequency: 2, direction: [1, 0, 0], gustLength: 2)
        let side = 60
        let field = try await tipDisplacements(side: side, spacing: 0.1, wind: wind, time: 1)

        // Some patches are pressed down while others stay nearly still.
        let flat = field.flatMap { $0 }
        let peak = flat.max()!
        let strong = flat.filter { $0 > wind.strength * 0.6 }.count
        let calm = flat.filter { $0 < wind.strength * 0.25 }.count
        #expect(strong > flat.count / 10, "too few blades caught in a gust: \(strong)")
        #expect(calm > flat.count / 10, "too few calm blades: \(calm)")

        // Clumps, not noise: neighbors 10 cm apart move almost the same, unlike per-blade randomness.
        var here: [Float] = [], neighbor: [Float] = []
        for row in 0..<side { for column in 0..<side - 1 { here.append(field[row][column]); neighbor.append(field[row][column + 1]) } }
        let neighborCorrelation = correlation(here, neighbor)
        #expect(neighborCorrelation > 0.8, "neighbor correlation \(neighborCorrelation)")

        // The clumps travel: a second later the pattern has moved on.
        let later = try await tipDisplacements(side: side, spacing: 0.1, wind: wind, time: 2).flatMap { $0 }
        let overTime = correlation(flat, later)
        #expect(overTime < 0.6, "pattern barely changed: \(overTime)")
        print("gust field: peak \(peak) strong \(strong) calm \(calm) neighbor r \(neighborCorrelation) over-time r \(overTime)")
    }

    @Test func emptyRootsMakeAnEmptyEntity() async throws {
        let grass = try await GrassFieldEntity(roots: [])
        #expect(grass.bladeCount == 0)
        #expect(grass.components[ModelComponent.self] == nil)
    }
}
