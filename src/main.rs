use std::env;
use std::fs;

use semantica::SemanticVec;

const DEFAULT_FILEPATH: &str = "db.semix"; // db: database; semix: semantic indexer.

const HELP_MSG: &str = r#"
-h      --help                  Prints the default help screen.
-f      --filepath  [PATH]      Changes the database filepath from the default.
"#;

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut filepath = DEFAULT_FILEPATH;

    if args.len() < 2 {
        panic!("No arguments specified.\n\nSupported arguments:\n{}", HELP_MSG);
    }

    let mut i = 1;
    while i < args.len() {
        match args[i].to_lowercase().as_str() {
            "-h" | "--help" => {
                println!("{}", HELP_MSG);
                return;
            },
            "-f" | "--filepath" => {
                i += 1;
                filepath = &args[i];
            },
            any => {
                panic!("Argument {} not recognised.\n\nSupported arguments:\n{}", any, HELP_MSG);
            }
        }

        i += 1;
    }
}
