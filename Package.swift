// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "RealityGrass",
    platforms: [
        .visionOS(.v2),
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "RealityGrass", targets: ["RealityGrass"]),
    ],
    targets: [
        // Structs shared between Swift and the Metal compute shaders.
        .target(name: "GrassShaderTypes"),
        .target(
            name: "RealityGrass",
            dependencies: ["GrassShaderTypes"],
            resources: [.process("Shaders")]
        ),
        .testTarget(name: "RealityGrassTests", dependencies: ["RealityGrass"]),
    ]
)
