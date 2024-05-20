use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

pub fn read_merge_seq(file_name: &PathBuf) -> Vec<u8> {
    let mut fna_seqs = Vec::<u8>::new();

    let file = File::open(file_name).unwrap();
    let mut reader = BufReader::new(file);

    // TODO: add support for other types of files
    let mut buf = String::new();
    while reader.read_line(&mut buf).unwrap() > 0 {
        if !buf.starts_with('>') {
            if buf.ends_with('\n') {
                buf.pop(); //remove \n character
            }
            if buf.ends_with('\r') {
                buf.pop();
            }
            fna_seqs.extend_from_slice(buf.as_bytes());
        } else {
            fna_seqs.push(b'N');
        }
        buf.clear();
    }

    fna_seqs
}

// use glob::glob;
// use indicatif::{ProgressBar, ProgressStyle};
// use rayon::prelude::*;
// use seq_io::fasta::Reader;

// pub fn parse_fasta_file(path_fna: &String, ksize: usize, scaled: u64) {
//     // get files
//     let files: Vec<_> = glob(Path::new(&path_fna).join("*.fna").to_str().unwrap())
//         .expect("Failed to read glob pattern")
//         .collect();

//     // progress bar
//     let pb = ProgressBar::new(files.len() as u64);
//     pb.set_style(
//         ProgressStyle::default_bar()
//             .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
//             .unwrap()
//             .progress_chars("##-"),
//     );

//     // create set
//     let index_vec: Vec<usize> = (0..files.len()).collect();
//     let a: Vec<Vec<u8>> = index_vec
//         .par_iter()
//         .map(|i| {
//             let mut fna_seqs = Vec::<u8>::with_capacity(10_000_000);

//             let mut fastx_reader = Reader::from_path(files[*i].as_ref().unwrap()).unwrap();
//             while let Some(record) = fastx_reader.next() {
//                 let seqrec = record.unwrap();
//                 let mut seq_i = seqrec.owned_seq();
//                 seq_i.push(b'N');
//                 fna_seqs.append(&mut seq_i);
//             }

//             pb.inc(1);
//             pb.eta();
//             fna_seqs[0..5000].to_vec()
//         })
//         .collect();

//     pb.finish();
// }

// pub fn parse_my_fasta_file(path_fna: &String, ksize: usize, scaled: u64) {
//     // get files
//     let files: Vec<_> = glob(Path::new(&path_fna).join("*.fna").to_str().unwrap())
//         .expect("Failed to read glob pattern")
//         .collect();

//     // progress bar
//     let pb = ProgressBar::new(files.len() as u64);
//     pb.set_style(
//         ProgressStyle::default_bar()
//             .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
//             .unwrap()
//             .progress_chars("##-"),
//     );

//     // create set
//     let index_vec: Vec<usize> = (0..files.len()).collect();
//     let a: Vec<Vec<u8>> = index_vec
//         .par_iter()
//         .map(|i| {
//             let fna_seqs = read_merge_seq(files[*i].as_ref().unwrap());

//             pb.inc(1);
//             pb.eta();
//             fna_seqs[0..200].to_vec()
//         })
//         .collect();

//     pb.finish();
// }
