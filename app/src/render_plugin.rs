use bevy_app::{App, Plugin, Startup, Update};
use bevy_ecs::{
    entity::Entity,
    query::With,
    system::{Commands, NonSend, ResMut, Resource, Single},
};
use bevy_window::{PrimaryWindow, RawHandleWrapper, Window};
use bevy_winit::WinitWindows;
use data::{
    camera::{CameraFov, CameraGpu},
    transform::Transform,
};
use renderer::state::VxState;

use crate::player_plugin::Player;

pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup).add_systems(Update, update);
    }
}

fn setup(
    mut commands: Commands,
    window: Single<(Entity, &Window)>,
    winit_windows: NonSend<WinitWindows>,
) {
    let (window_entity, window) = window.into_inner();

    let winit_window = winit_windows.get_window(window_entity).unwrap();
    let wrapper = RawHandleWrapper::new(winit_window).unwrap();

    let display_handle = wrapper.display_handle;
    let window_handle = wrapper.window_handle;

    commands.entity(window_entity).insert(wrapper);

    commands.insert_resource(RenderState(
        VxState::new(
            "Hello",
            1,
            window.width(),
            window.height(),
            display_handle,
            window_handle,
        )
        .unwrap(),
    ));
}

fn update(
    mut render_state: ResMut<RenderState>,
    window: Single<&Window, With<PrimaryWindow>>,
    player: Single<(&Transform, &CameraFov), With<Player>>,
) {
    let (transform, fov) = player.into_inner();
    render_state
        .0
        .draw_frame(
            window.width(),
            window.height(),
            CameraGpu::new(transform, fov.degrees(), window.width(), window.height()),
        )
        .unwrap();
}

#[derive(Resource)]
pub struct RenderState(pub VxState);
