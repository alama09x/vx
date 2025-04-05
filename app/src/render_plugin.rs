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
    acceleration_structure_state::AccelerationStructureState, buffer_state::BufferState,
    command_state::CommandState, init_state::InitState, pipeline_state::PipelineState,
    swapchain_state::SwapchainState, CurrentFrame,
};

use crate::player_plugin::Player;

pub struct RenderPlugin;

#[derive(Event)]
pub struct CleanupEvent;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<CleanupEvent>()
            .init_resource::<CurrentFrame>()
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

    let swapchain_state =
        SwapchainState::new(&init_state, Vec2::new(window.width(), window.height())).unwrap();

    let pipeline_state = PipelineState::new(&init_state).unwrap();

    let buffer_state = BufferState::new(&init_state).unwrap();

    let acceleration_structure_state = AccelerationStructureState::new(
        &init_state,
        &swapchain_state,
        &pipeline_state,
        &buffer_state,
    )
    .unwrap();

    let command_state = CommandState::new(&init_state).unwrap();

    commands.insert_resource(init_state);
    commands.insert_resource(swapchain_state);
    commands.insert_resource(pipeline_state);
    commands.insert_resource(buffer_state);
    commands.insert_resource(acceleration_structure_state);
    commands.insert_resource(command_state);
}

fn update(
    init_state: Res<InitState>,
    mut swapchain_state: ResMut<SwapchainState>,
    mut buffer_state: ResMut<BufferState<'static>>,
    pipeline_state: Res<PipelineState<'static>>,
    mut acceleration_structure_state: ResMut<AccelerationStructureState<'static>>,
    mut command_state: ResMut<CommandState>,
    mut current_frame: ResMut<CurrentFrame>,
    window: Single<&Window, With<PrimaryWindow>>,
    player: Single<(&Transform, &CameraFov), With<Player>>,
) {
    let (transform, fov) = player.into_inner();
    command_state
        .draw_frame(
            &init_state,
            &mut swapchain_state,
            &pipeline_state,
            &mut buffer_state,
            &mut acceleration_structure_state,
            Vec2::new(window.width(), window.height()),
            CameraGpu::new(transform, fov.degrees(), window.width(), window.height()),
            current_frame.0,
        )
        .unwrap();
    current_frame.0 = current_frame.next();
}

fn cleanup(
    mut cleanup_reader: EventReader<CleanupEvent>,
    init_state: Res<InitState>,
    swapchain_state: Res<SwapchainState>,
    mut buffer_state: ResMut<BufferState<'static>>,
    mut pipeline_state: ResMut<PipelineState<'static>>,
    mut acceleration_structure_state: ResMut<AccelerationStructureState<'static>>,
    command_state: Res<CommandState>,
) {
    for _ in cleanup_reader.read() {
        println!("Goodbye!");
        init_state.wait_idle().unwrap();
        command_state.cleanup(&init_state);
        acceleration_structure_state.cleanup(&init_state);
        buffer_state.cleanup(&init_state);
        pipeline_state.cleanup(&init_state);
        swapchain_state.cleanup(&init_state);
    }
}
