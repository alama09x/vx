use std::time::{Duration, Instant};

use bevy_app::{Plugin, Update};
use bevy_ecs::system::{ResMut, Resource};

pub struct TimePlugin;

impl Plugin for TimePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_resource::<Time>().add_systems(Update, update_time);
    }
}

#[derive(Resource, Clone, Copy)]
pub struct Time {
    start: Instant,
    last: Instant,
    current: Instant,
}

impl Default for Time {
    fn default() -> Self {
        Self {
            start: Instant::now(),
            last: Instant::now(),
            current: Instant::now(),
        }
    }
}

impl Time {
    pub fn delta(&self) -> Duration {
        self.current.duration_since(self.last)
    }

    pub fn delta_secs(&self) -> f32 {
        self.delta().as_secs_f32()
    }

    pub fn elapsed(&self) -> Duration {
        self.current.duration_since(self.start)
    }

    pub fn elapsed_secs(&self) -> f32 {
        self.elapsed().as_secs_f32()
    }
}

fn update_time(mut time: ResMut<Time>) {
    time.last = time.current;
    time.current = Instant::now();
}
