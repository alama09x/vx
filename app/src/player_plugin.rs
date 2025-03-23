use std::f32;

use bevy_app::{Plugin, Startup, Update};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    event::EventReader,
    query::With,
    schedule::IntoSystemConfigs,
    system::{Commands, Res, ResMut, Resource, Single},
};
use bevy_input::{
    keyboard::KeyCode,
    mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    ButtonInput,
};
use bevy_window::{PrimaryWindow, WindowFocused};
use data::{camera::CameraFov, transform::Transform};
use glam::{EulerRot, Quat, Vec3};

use crate::time_plugin::Time;

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_resource::<IgnoreNextDelta>()
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    move_player,
                    (ignore_deltas, rotate_player).chain(),
                    zoom_player,
                ),
            );
    }
}

#[derive(Component, Clone, Copy)]
pub struct Player;

#[derive(Resource)]
pub struct IgnoreNextDelta(bool);

impl Default for IgnoreNextDelta {
    fn default() -> Self {
        Self(true)
    }
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Player,
        CameraFov::from_degrees(45.0),
        Transform::from_xyz(0.0, 0.0, 16.0),
    ));
}

const MOVE_SPEED: f32 = 5.0;

const YAW_SPEED: f32 = 0.5;
const PITCH_SPEED: f32 = 0.5;

const PITCH_LIMIT: f32 = f32::consts::FRAC_PI_2 - 0.01;

const SCROLL_SPEED: f32 = 10.0;

pub fn move_player(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    transform: Single<&mut Transform, With<Player>>,
) {
    let mut transform = transform.into_inner();

    let speed = MOVE_SPEED * time.delta_secs();

    let remove_y = Vec3::X + Vec3::Z;
    let local_x = (transform.rotation * Vec3::X * remove_y).normalize() * speed;
    let local_z = (transform.rotation * Vec3::Z * remove_y).normalize() * speed;

    for key in keys.get_pressed() {
        match key {
            KeyCode::KeyW => transform.translation -= local_z,
            KeyCode::KeyA => transform.translation -= local_x,
            KeyCode::KeyS => transform.translation += local_z,
            KeyCode::KeyD => transform.translation += local_x,
            KeyCode::Space => transform.translation.y -= speed,
            KeyCode::ShiftLeft => transform.translation.y += speed,
            _ => (),
        }
    }
}

pub fn ignore_deltas(
    mut ignore_next_delta: ResMut<IgnoreNextDelta>,
    mut window_focused_reader: EventReader<WindowFocused>,
    primary_window: Single<Entity, With<PrimaryWindow>>,
) {
    for window_focused in window_focused_reader.read() {
        if window_focused.window == *primary_window && window_focused.focused {
            ignore_next_delta.0 = true;
        }
    }
}

pub fn rotate_player(
    time: Res<Time>,
    mut mouse_motion: ResMut<AccumulatedMouseMotion>,
    mut ignore_next_delta: ResMut<IgnoreNextDelta>,
    transform: Single<&mut Transform, With<Player>>,
) {
    if mouse_motion.delta.x == 0.0 && mouse_motion.delta.y == 0.0 {
        return;
    }

    if ignore_next_delta.0 {
        ignore_next_delta.0 = false;
        mouse_motion.delta.x = 0.0;
        mouse_motion.delta.y = 0.0;
        return;
    }

    let delta_time = time.delta_secs();
    let mut transform = transform.into_inner();

    let delta = mouse_motion.delta;

    let dyaw = delta.x * YAW_SPEED * delta_time;
    let dpitch = -delta.y * PITCH_SPEED * delta_time;

    let (yaw, pitch, _roll) = transform.rotation.to_euler(EulerRot::YXZ);
    let yaw = yaw - dyaw;
    let pitch = (pitch - dpitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);

    transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
}

pub fn zoom_player(
    time: Res<Time>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    player: Single<&mut CameraFov, With<Player>>,
) {
    let mut fov = player.into_inner();
    fov.zoom(mouse_scroll.delta.y, SCROLL_SPEED * time.delta_secs());
}
