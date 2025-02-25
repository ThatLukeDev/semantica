use semantica::SemanticVec;

fn main() {
    let mut db: SemanticVec<i32> = SemanticVec::new();

    db.add("One", 1);
    db.add("Two", 2);
    db.add("Three", 3);
    db.add("Four", 4);
    db.add("Five", 5);

    db.add("Dog", 0);
    db.add("Cat", -1);

    match db.search("uno") {
        Some(item) => println!("{}", item),
        None => ()
    }
}
