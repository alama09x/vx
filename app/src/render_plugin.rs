use bevy_app::{App, Last, Plugin, Startup, Update};
use bevy_ecs::{
    entity::Entity,
    event::{Event, EventReader},
    query::With,
    system::{Commands, NonSend, Res, ResMut, Single},
};
use bevy_window::{PrimaryWindow, RawHandleWrapper, Window};
use bevy_winit::WinitWindows;
use data::{
    camera::{CameraFov, CameraGpu},
    transform::Transform,
};
use glam::Vec2;
use renderer::{
    AccelerationStructureState, BuffersState, CommandSyncState, InitState, PipelineState,
    SwapchainRenderState,
};

use crate::player_plugin::Player;

pub struct RenderPlugin;

#[derive(Event)]
pub struct CleanupEvent;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<CleanupEvent>()
            .add_systems(Startup, setup)
            .add_systems(Update, update)
            .add_systems(Last, cleanup);
    }
}

fn setup(
    mut commands: Commands,
    window: Single<(Entity, &Window), With<PrimaryWindow>>,
    winit_windows: NonSend<WinitWindows>,
) {
    let (window_entity, window) = window.into_inner();

    let winit_window = winit_windows.get_window(window_entity).unwrap();
    let wrapper = RawHandleWrapper::new(winit_window).unwrap();

    let display_handle = wrapper.display_handle;
    let window_handle = wrapper.window_handle;

    commands.entity(window_entity).insert(wrapper);

    let init_state = InitState::new("Hello", 1, display_handle, window_handle).unwrap();

    let swapchain_render_state =
        SwapchainRenderState::new(&init_state, Vec2::new(window.width(), window.height())).unwrap();

    let pipeline_state = PipelineState::new(&init_state).unwrap();

    let buffers_state = BuffersState::new(&init_state).unwrap();

    let acceleration_structure_state = AccelerationStructureState::new(
        &init_state,
        &swapchain_render_state,
        &pipeline_state,
        &buffers_state,
    )
    .unwrap();

    let command_sync_state = CommandSyncState::new(&init_state).unwrap();

    commands.insert_resource(init_state);
    commands.insert_resource(swapchain_render_state);
    commands.insert_resource(pipeline_state);
    commands.insert_resource(buffers_state);
    commands.insert_resource(acceleration_structure_state);
    commands.insert_resource(command_sync_state);
}

fn update(
    init_state: Res<InitState>,
    mut swapchain_render_state: ResMut<SwapchainRenderState>,
    mut buffers_state: ResMut<BuffersState>,
    pipeline_state: Res<PipelineState>,
    acceleration_structure_state: Res<AccelerationStructureState>,
    mut command_sync_state: ResMut<CommandSyncState>,
    window: Single<&Window, With<PrimaryWindow>>,
    player: Single<(&Transform, &CameraFov), With<Player>>,
) {
    let (transform, fov) = player.into_inner();
    command_sync_state
        .draw_frame(
            &init_state,
            &mut swapchain_render_state,
            &pipeline_state,
            &mut buffers_state,
            &acceleration_structure_state,
            Vec2::new(window.width(), window.height()),
            CameraGpu::new(transform, fov.degrees(), window.width(), window.height()),
        )
        .unwrap();
}

fn cleanup(
    mut cleanup_reader: EventReader<CleanupEvent>,
    init_state: Res<InitState>,
    swapchain_render_state: Res<SwapchainRenderState>,
    buffers_state: Res<BuffersState>,
    pipeline_state: Res<PipelineState>,
    acceleration_structure_state: Res<AccelerationStructureState>,
    command_sync_state: Res<CommandSyncState>,
) {
    for _ in cleanup_reader.read() {
        init_state.wait_idle().unwrap();
        command_sync_state.cleanup(&init_state);
        acceleration_structure_state.cleanup(&init_state);
        buffers_state.cleanup(&init_state);
        pipeline_state.cleanup(&init_state);
        swapchain_render_state.cleanup(&init_state);
    }
}
