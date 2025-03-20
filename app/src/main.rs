use app::VxApplication;
use winit::event_loop::EventLoop;

fn main() {
    // let vert_path = Path::new("./bin/shader.vert.spv");
    // let frag_path = Path::new("./bin/shader.frag.spv");
    // if !vert_path.exists() || !frag_path.exists() {
    //     Command::new("python ./compile_shaders.py ./shaders/shader.vert")
    //         .output()
    //         .unwrap();
    //     Command::new("python ./compile_shaders.py ./shaders/shader.vert")
    //         .output()
    //         .unwrap();
    // }

    let mut app = VxApplication::new("Hello World", 0, "Hello World", 800.0, 600.0);
    let event_loop = EventLoop::new().unwrap();

    event_loop.run_app(&mut app).unwrap();
}
