import Foundation
import Metal

/// Errors thrown while building a grass field.
public enum GrassError: Error {
    case metalUnavailable
    case shaderLibraryMissing
    case bufferAllocationFailed
    case commandEncodingFailed
}

/// The Metal device, queue and pipelines shared by every grass field.
@MainActor
final class GrassGPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let generatePipeline: MTLComputePipelineState
    let updatePipeline: MTLComputePipelineState

    private static var cached: GrassGPU?

    static func shared() throws -> GrassGPU {
        if let cached { return cached }
        let gpu = try GrassGPU()
        cached = gpu
        return gpu
    }

    private init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw GrassError.metalUnavailable
        }
        queue.label = "RealityGrass"
        guard let library = try? device.makeDefaultLibrary(bundle: .module),
              let generate = library.makeFunction(name: "generateGrassBlades"),
              let update = library.makeFunction(name: "updateGrassVertices") else {
            throw GrassError.shaderLibraryMissing
        }
        self.device = device
        self.queue = queue
        self.generatePipeline = try device.makeComputePipelineState(function: generate)
        self.updatePipeline = try device.makeComputePipelineState(function: update)
    }

    /// Threadgroup counts that cover `count` threads in one dimension.
    static func grid(for count: Int, pipeline: MTLComputePipelineState) -> (groups: MTLSize, perGroup: MTLSize) {
        let width = pipeline.threadExecutionWidth
        return (MTLSize(width: (count + width - 1) / width, height: 1, depth: 1),
                MTLSize(width: width, height: 1, depth: 1))
    }
}
