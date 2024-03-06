use hyper_gen_rust::{dist, params, sketch, types, utils};


fn main() {
    let cli_params = utils::create_cli();

    rayon::ThreadPoolBuilder::new()
        .num_threads(cli_params.threads as usize)
        .build_global()
        .unwrap();

    if cli_params.mode == params::CMD_SKETCH {
        let sketch_params = types::SketchParams::new(&cli_params);
        sketch::sketch(sketch_params);
    } else if cli_params.mode == params::CMD_DIST {
        let mut sketch_dist = types::SketchDist::new(&cli_params);
        dist::dist(&mut sketch_dist);
    } else if cli_params.mode == params::CMD_SEARCH {
        let sketch_dist = types::SketchDist::new(&cli_params);
        utils::dump_sketch_to_txt(&sketch_dist.path_ref_sketch.as_path());
    }
}
