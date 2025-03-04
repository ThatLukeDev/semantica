# Semantica

A semantic search engine written in Rust.

---

Command line arguments:
```
-h      --help                              Prints the default help screen.
-f      --filepath  [PATH]                  Changes the database filepath from the default.
-s      --search    [NAME]                  Returns the index from the specified name embeddings, or none if not found.
-x      --remove    [VAL1] [VAL2?]          Removes a value from the semix.
-a      --add       [NAME1] [VAL1]          Adds a value into the semix.
                    [NAME2?] [VAL2?] ...    (VAL_ should be any positive integer index)
```

![Semantic search console example](https://github.com/user-attachments/assets/45d9a664-092f-4ff4-874e-fae83897401d)
