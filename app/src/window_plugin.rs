use bevy_app::{AppExit, Plugin, Update};
use bevy_ecs::{
    entity::Entity,
    event::{EventReader, EventWriter},
    query::With,
    system::{Res, ResMut, Single},
};
use bevy_input::{keyboard::KeyCode, ButtonInput};
use bevy_window::{CursorGrabMode, PrimaryWindow, Window, WindowFocused, WindowResized};
use glam::Vec2;
use renderer::{
    acceleration_structure_state::AccelerationStructureState, buffer_state::BufferState,
    init_state::InitState, swapchain_state::SwapchainState,
};

use crate::render_plugin::CleanupEvent;

pub struct WindowPlugin;

impl Plugin for WindowPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(
            Update,
            (
                close_window_on_escape,
                grab_cursor_at_center,
                recreate_swapchain,
            ),
        );
    }
}

fn close_window_on_escape(
    keys: Res<ButtonInput<KeyCode>>,
    mut cleanup_writer: EventWriter<CleanupEvent>,
    mut exit_writer: EventWriter<AppExit>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        cleanup_writer.send(CleanupEvent);
        exit_writer.send(AppExit::Success);
    }
}

fn grab_cursor_at_center(
    mut focus_reader: EventReader<WindowFocused>,
    window: Single<(Entity, &mut Window), With<PrimaryWindow>>,
) {
    let (window_entity, mut window) = window.into_inner();
    let half_size = window.size() * 0.5;
    for focus in focus_reader.read() {
        if focus.window == window_entity {
            if focus.focused {
                window.cursor_options.grab_mode = CursorGrabMode::Confined;
                window.set_cursor_position(Some(half_size));
            } else {
                window.cursor_options.grab_mode = CursorGrabMode::None;
            }
        }
    }
}

fn recreate_swapchain(
    mut resized_reader: EventReader<WindowResized>,
    init_state: Res<InitState>,
    mut swapchain_state: ResMut<SwapchainState>,
    buffer_state: Res<BufferState<'static>>,
    mut acceleration_structure_state: ResMut<AccelerationStructureState<'static>>,
) {
    for resize in resized_reader.read() {
        swapchain_state
            .recreate_swapchain(
                &init_state,
                &buffer_state,
                &mut acceleration_structure_state,
                Vec2::new(resize.width, resize.height),
            )
            .unwrap();
    }
}
