import GrassShaderTypes
import Metal
import RealityKit

/// A field of GPU-generated grass blades that sway in the wind.
///
/// Blade geometry is built once by a compute shader and animated every frame by a second
/// compute shader that writes straight into a `LowLevelMesh`, so the CPU never touches
/// per-vertex data.
///
/// ```swift
/// let ground = GroundSample.grid(size: [4, 4], resolution: [64, 64]) { x, z in
///     0.1 * sin(x) * cos(z)
/// }
/// let roots = GrassScatter(density: 600).roots(on: ground)
/// let grass = try await GrassFieldEntity(roots: roots)
/// content.add(grass)
/// ```
@MainActor
public final class GrassFieldEntity: Entity {
    /// The wind applied to this field. Changes take effect on the next frame.
    public var wind: GrassWind {
        get { components[GrassComponent.self]?.wind ?? .calm }
        set { components[GrassComponent.self]?.wind = newValue }
    }

    /// How often the field animates, depending on its distance from `viewerPosition`.
    public var lod: GrassLOD {
        get { components[GrassComponent.self]?.lod ?? .disabled }
        set { components[GrassComponent.self]?.lod = newValue }
    }

    /// The viewer's position in world space, used for LOD. While `nil`, the field animates every frame.
    ///
    /// On visionOS you can get it from `WorldTrackingProvider.queryDeviceAnchor(atTimestamp:)`.
    public var viewerPosition: SIMD3<Float>? {
        get { components[GrassComponent.self]?.viewerPosition }
        set { components[GrassComponent.self]?.viewerPosition = newValue }
    }

    /// The number of blades in the field.
    public var bladeCount: Int {
        Int(components[GrassComponent.self]?.state.bladeCount ?? 0)
    }

    /// Creates a grass field with one blade rooted at each position.
    ///
    /// - Parameters:
    ///   - roots: Blade root positions in the entity's local space, Y up. See `GrassScatter`.
    ///   - shape: The size and curl of the blades.
    ///   - wind: The initial wind.
    ///   - lod: How animation frequency drops with distance.
    ///   - material: The blade material. UV `v` runs from the root (0) to the tip (1), and UV `u` is a
    ///     random value per blade. Defaults to `GrassMaterial.make()`.
    public init(
        roots: [SIMD3<Float>],
        shape: GrassBladeShape = .init(),
        wind: GrassWind = .breeze,
        lod: GrassLOD = .default,
        material: (any Material)? = nil
    ) async throws {
        super.init()
        GrassAnimationSystem.registerIfNeeded()
        guard !roots.isEmpty else { return }

        let gpu = try GrassGPU.shared()
        let bladeCount = roots.count
        let mesh = try Self.makeMesh(bladeCount: bladeCount, bounds: Self.bounds(of: roots, shape: shape, wind: wind))

        guard let restVertices = gpu.device.makeBuffer(
            length: MemoryLayout<GrassVertex>.stride * bladeCount * Int(GRASS_VERTICES_PER_BLADE),
            options: .storageModePrivate
        ), let rootBuffer = gpu.device.makeBuffer(
            bytes: roots,
            length: MemoryLayout<SIMD3<Float>>.stride * bladeCount,
            options: .storageModeShared
        ) else {
            throw GrassError.bufferAllocationFailed
        }

        let state = GrassFieldState(mesh: mesh, restVertices: restVertices, bladeCount: UInt32(bladeCount), shape: shape)
        var params = state.params(wind: wind, time: 0)

        guard let commandBuffer = gpu.queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GrassError.commandEncodingFailed
        }
        // Build the rest pose and indices, then pose the mesh once so it is visible
        // before the first animation frame.
        encoder.setComputePipelineState(gpu.generatePipeline)
        encoder.setBuffer(restVertices, offset: 0, index: 0)
        encoder.setBuffer(mesh.replaceIndices(using: commandBuffer), offset: 0, index: 1)
        encoder.setBuffer(rootBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<GrassParams>.stride, index: 3)
        let generateGrid = GrassGPU.grid(for: bladeCount, pipeline: gpu.generatePipeline)
        encoder.dispatchThreadgroups(generateGrid.groups, threadsPerThreadgroup: generateGrid.perGroup)

        state.encodeAnimation(into: encoder, commandBuffer: commandBuffer, params: &params, gpu: gpu)
        encoder.endEncoding()

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            commandBuffer.addCompletedHandler { _ in continuation.resume() }
            commandBuffer.commit()
        }

        let meshResource = try await MeshResource(from: mesh)
        let resolvedMaterial: any Material
        if let material {
            resolvedMaterial = material
        } else {
            resolvedMaterial = try await GrassMaterial.make()
        }
        components.set(ModelComponent(mesh: meshResource, materials: [resolvedMaterial]))
        components.set(GrassComponent(state: state, wind: wind, lod: lod))
    }

    /// Creates an empty entity with no grass.
    public required init() {
        super.init()
    }

    private static func makeMesh(bladeCount: Int, bounds: BoundingBox) throws -> LowLevelMesh {
        let descriptor = LowLevelMesh.Descriptor(
            vertexCapacity: bladeCount * Int(GRASS_VERTICES_PER_BLADE),
            vertexAttributes: [
                .init(semantic: .position, format: .float3, offset: MemoryLayout<GrassVertex>.offset(of: \.position)!),
                .init(semantic: .normal, format: .float3, offset: MemoryLayout<GrassVertex>.offset(of: \.normal)!),
                .init(semantic: .uv0, format: .float2, offset: MemoryLayout<GrassVertex>.offset(of: \.uv)!),
            ],
            vertexLayouts: [.init(bufferIndex: 0, bufferStride: MemoryLayout<GrassVertex>.stride)],
            indexCapacity: bladeCount * Int(GRASS_INDICES_PER_BLADE)
        )
        let mesh = try LowLevelMesh(descriptor: descriptor)
        mesh.parts.replaceAll([
            LowLevelMesh.Part(indexCount: descriptor.indexCapacity, topology: .triangle, bounds: bounds),
        ])
        return mesh
    }

    /// Bounds that contain every blade at full height, plus room to sway.
    static func bounds(of roots: [SIMD3<Float>], shape: GrassBladeShape, wind: GrassWind) -> BoundingBox {
        var lower = SIMD3<Float>(repeating: .infinity)
        var upper = SIMD3<Float>(repeating: -.infinity)
        for root in roots {
            lower = simd_min(lower, root)
            upper = simd_max(upper, root)
        }
        // Patch and per-blade variation make a blade at most 1.3 × 1.35 ≈ 1.8 times the
        // nominal height. A blade bending by any angle stays within one blade length of its root.
        let reach = shape.height * 1.8 + shape.width
        return BoundingBox(min: lower - [reach, 0, reach], max: upper + [reach, reach, reach])
    }
}

/// Per-field GPU resources.
@MainActor
final class GrassFieldState {
    let mesh: LowLevelMesh
    let restVertices: MTLBuffer
    let bladeCount: UInt32
    let shape: GrassBladeShape
    var time: Float = 0
    var framesSinceUpdate = 0

    init(mesh: LowLevelMesh, restVertices: MTLBuffer, bladeCount: UInt32, shape: GrassBladeShape) {
        self.mesh = mesh
        self.restVertices = restVertices
        self.bladeCount = bladeCount
        self.shape = shape
    }

    func params(wind: GrassWind, time: Float) -> GrassParams {
        GrassParams(
            bladeCount: bladeCount,
            bladeHeight: shape.height,
            bladeWidth: shape.width,
            bladeCurvature: shape.curvature,
            time: time,
            windStrength: wind.strength,
            windFrequency: wind.frequency,
            windGustLength: max(wind.gustLength, 0.01),
            windDirection: wind.horizontalDirection
        )
    }

    func encodeAnimation(into encoder: MTLComputeCommandEncoder, commandBuffer: MTLCommandBuffer, params: inout GrassParams, gpu: GrassGPU) {
        encoder.setComputePipelineState(gpu.updatePipeline)
        encoder.setBuffer(restVertices, offset: 0, index: 0)
        encoder.setBuffer(mesh.replace(bufferIndex: 0, using: commandBuffer), offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<GrassParams>.stride, index: 2)
        let grid = GrassGPU.grid(for: Int(bladeCount) * Int(GRASS_VERTICES_PER_BLADE), pipeline: gpu.updatePipeline)
        encoder.dispatchThreadgroups(grid.groups, threadsPerThreadgroup: grid.perGroup)
    }
}

/// Marks an entity as an animated grass field. Added by `GrassFieldEntity`.
struct GrassComponent: Component {
    let state: GrassFieldState
    var wind: GrassWind
    var lod: GrassLOD
    var viewerPosition: SIMD3<Float>?
}

/// Animates every `GrassComponent` in the scene, batching all fields into one command buffer per frame.
@MainActor
final class GrassAnimationSystem: System {
    private static let query = EntityQuery(where: .has(GrassComponent.self))
    private static var isRegistered = false

    static func registerIfNeeded() {
        guard !isRegistered else { return }
        isRegistered = true
        GrassComponent.registerComponent()
        registerSystem()
    }

    init(scene: Scene) {}

    func update(context: SceneUpdateContext) {
        guard let gpu = try? GrassGPU.shared() else { return }
        let deltaTime = Float(context.deltaTime)
        var commandBuffer: MTLCommandBuffer?
        var encoder: MTLComputeCommandEncoder?

        for entity in context.entities(matching: Self.query, updatingSystemWhen: .rendering) {
            guard let component = entity.components[GrassComponent.self] else { continue }
            let state = component.state
            // Time advances every frame so skipped frames do not slow the animation down.
            state.time += deltaTime
            state.framesSinceUpdate += 1

            let interval = component.viewerPosition.map {
                component.lod.updateInterval(atDistance: simd_distance($0, entity.position(relativeTo: nil)))
            } ?? 1
            guard state.framesSinceUpdate >= interval else { continue }
            state.framesSinceUpdate = 0

            if encoder == nil {
                commandBuffer = gpu.queue.makeCommandBuffer()
                encoder = commandBuffer?.makeComputeCommandEncoder()
            }
            guard let commandBuffer, let encoder else { return }
            var params = state.params(wind: component.wind, time: state.time)
            state.encodeAnimation(into: encoder, commandBuffer: commandBuffer, params: &params, gpu: gpu)
        }

        encoder?.endEncoding()
        commandBuffer?.commit()
    }
}
