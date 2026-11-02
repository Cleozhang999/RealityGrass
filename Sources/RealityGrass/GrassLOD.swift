/// Animates distant grass less often to save GPU time.
///
/// Distance is measured from the viewer to the grass field's origin, so the whole field
/// uses one level. To get finer-grained LOD over a large area, split it into several
/// smaller `GrassFieldEntity` tiles.
public struct GrassLOD: Sendable, Equatable {
    public struct Level: Sendable, Equatable {
        /// The level applies up to this distance from the viewer, in meters.
        public var maxDistance: Float
        /// Animate every N-th frame. 1 means every frame.
        public var updateInterval: Int

        public init(maxDistance: Float, updateInterval: Int) {
            precondition(updateInterval >= 1, "updateInterval must be at least 1")
            self.maxDistance = maxDistance
            self.updateInterval = updateInterval
        }
    }

    /// Levels sorted by increasing `maxDistance`.
    public var levels: [Level]

    public init(levels: [Level]) {
        self.levels = levels.sorted { $0.maxDistance < $1.maxDistance }
    }

    /// Animates every frame regardless of distance.
    public static let disabled = GrassLOD(levels: [])

    public static let `default` = GrassLOD(levels: [
        Level(maxDistance: 5, updateInterval: 1),
        Level(maxDistance: 10, updateInterval: 2),
        Level(maxDistance: 20, updateInterval: 3),
        Level(maxDistance: 30, updateInterval: 4),
        Level(maxDistance: .infinity, updateInterval: 6),
    ])

    /// How often to animate a field at `distance` meters. Beyond the last level, the
    /// last level's interval applies.
    public func updateInterval(atDistance distance: Float) -> Int {
        levels.first { distance <= $0.maxDistance }?.updateInterval ?? levels.last?.updateInterval ?? 1
    }
}
