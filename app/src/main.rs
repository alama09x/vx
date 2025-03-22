use app::{
    player_plugin::PlayerPlugin, render_plugin::RenderPlugin, time_plugin::TimePlugin,
    window_plugin,
};
use bevy_a11y::AccessibilityPlugin;
use bevy_app::App;
use bevy_ecs::event::Event;
use bevy_input::InputPlugin;
use bevy_window::{CursorGrabMode, CursorOptions, Window, WindowPlugin, WindowResolution};
use bevy_winit::WinitPlugin;

fn main() {
    // let mut app = VxApplication::new("Hello World", 0, "Hello World", 800, 600);
    // let event_loop = EventLoop::new().unwrap();

    // event_loop.run_app(&mut app).unwrap();
    App::new()
        .add_plugins((
            AccessibilityPlugin,
            InputPlugin,
            WinitPlugin::<WinitEvent>::default(),
            WindowPlugin {
                primary_window: Some(Window {
                    cursor_options: CursorOptions {
                        visible: false,
                        grab_mode: CursorGrabMode::Locked,
                        ..Default::default()
                    },
                    resolution: WindowResolution::new(800.0, 600.0),
                    ..Default::default()
                }),
                close_when_requested: true,
                ..Default::default()
            },
            window_plugin::WindowPlugin,
            TimePlugin,
            RenderPlugin,
            PlayerPlugin,
        ))
        .run();
}

#[derive(Event, Default)]
struct WinitEvent;
