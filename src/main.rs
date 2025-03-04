use std::env;
use std::fs;
use std::io::Write;
use std::path;

use semantica::SemanticVec;
use semantica::byte_conversion::ByteConversion;

const DEFAULT_FILEPATH: &str = "db.semix"; // db: database; semix: semantic indexer.

const HELP_MSG: &str = r#"
-h      --help                              Prints the default help screen.
-f      --filepath  [PATH]                  Changes the database filepath from the default.
-s      --search    [NAME]                  Returns the index from the specified name embeddings, or none if not found.
-x      --remove    [VAL1] [VAL2?]          Removes a value from the semix.
-a      --add       [NAME1] [VAL1]          Adds a value into the semix.
                    [NAME2?] [VAL2?] ...    (VAL_ should be any positive integer index)
"#;

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut filepath = DEFAULT_FILEPATH;

    if args.len() < 2 {
        panic!("No arguments specified.\n\nSupported arguments:\n{}", HELP_MSG);
    }

    let mut search: Option<&str> = None;
    let mut delete: Option<Vec<usize>> = None;
    let mut append: Vec<(&str, usize)> = vec![];

    let mut i = 1;
    while i < args.len() {
        match args[i].to_lowercase().as_str() {
            "-h" | "--help" => {
                panic!("{}", HELP_MSG);
            },
            "-f" | "--filepath" => {
                i += 1;
                filepath = &args[i];
            },
            "-s" | "--search" => {
                i += 1;
                search = Some(&args[i]);
            },
            "-x" | "--remove" => {
                i += 1;

                let mut removals = vec![];

                while i < args.len() {
                    removals.push(str::parse::<usize>(&args[i]).unwrap());

                    i += 1;
                }

                delete = Some(removals);
            },
            "-a" | "--add" => {
                i += 1;

                while i < args.len() {
                    append.push((&args[i], str::parse::<usize>(&args[i + 1]).unwrap()));

                    i += 2;
                }
            }
            any => {
                panic!("Argument {} not recognised.\n\nSupported arguments:\n{}", any, HELP_MSG);
            }
        }

        i += 1;
    }

    let mut db = match path::Path::new(filepath).exists() {
        true => {
            SemanticVec::<usize>::from_bytes(fs::read(filepath).unwrap())
        },
        false => {
            SemanticVec::<usize>::new()
        },
    };

    match search {
        Some(val) => {
            let result = db.search(val);

            match result {
                Some(res) => println!("{}", res),
                None => println!("null")
            }
        },
        None => {
            match delete {
                Some(res) => {
                    for id in res {
                        db.remove(id);
                    }
                }
                None => {
                    for pair in append {
                        db.add(pair.0, pair.1);
                    }
                }
            }

            let mut file = fs::File::create(filepath).unwrap();
            file.write_all(&db.to_bytes()).unwrap();
        }
    }
}
