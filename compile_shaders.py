import os
import subprocess
import sys

def compile_shader(shader_path):
    output_path = shader_path.replace('shaders', 'bin') + ".spv"

    command = ['glslc', shader_path, '-o', output_path]

    try:
        subprocess.run(command, check=True)
        print(f'Compiled {shader_path} to {output_path}')
    except subprocess.CalledProcessError as e:
        print(f"Error compiling {shader_path}: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: compile_shaders.py <shader1> <shader2> ...')
        sys.exit(1)

    for shader in sys.argv[1:]:
        if os.path.isfile(shader):
            compile_shader(shader)
        else:
            print(f"File not found: {shader}")