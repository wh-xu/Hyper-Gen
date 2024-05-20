use hyper_gen::{dist, params, sketch, sketch_cuda, types, utils};

fn main() {
    let cli_params = utils::create_cli();

    rayon::ThreadPoolBuilder::new()
        .num_threads(cli_params.threads as usize)
        .build_global()
        .unwrap();

    if cli_params.mode == params::CMD_SKETCH {
        let sketch_params = types::SketchParams::new(&cli_params);

        if sketch_params.device == "gpu" {
            sketch_cuda::sketch_cuda(sketch_params);
        } else {
            sketch::sketch(sketch_params);
        }
    } else if cli_params.mode == params::CMD_DIST {
        let mut sketch_dist = types::SketchDist::new(&cli_params);
        dist::dist(&mut sketch_dist);
    } else if cli_params.mode == params::CMD_SEARCH {
        // TODO: support for search
    }
}
