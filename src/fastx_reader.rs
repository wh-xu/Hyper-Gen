use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

// Read merged sequences from a genome file into single u8 vector
pub fn read_merge_seq(file_name: &PathBuf) -> Vec<u8> {
    let mut fna_seqs = Vec::<u8>::new();

    let file = File::open(file_name).unwrap();
    let mut reader = BufReader::new(file);

    let mut buf = String::new();
    while reader.read_line(&mut buf).unwrap() > 0 {
        if !buf.starts_with('>') {
            if buf.ends_with('\n') {
                buf.pop(); // remove \n character
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
