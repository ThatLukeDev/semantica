use rust_bert::pipelines::sentence_embeddings;

mod vector;

use crate::vector::Dot;

fn main() {
    let model = sentence_embeddings::SentenceEmbeddingsBuilder::remote(sentence_embeddings::SentenceEmbeddingsModelType::AllMiniLmL6V2).create_model()
        .unwrap();

    let input = [
        "Green cat",
        "Red cat",
        "Green dog",
        "Red dog",
    ];

    let embeddings: Vec<Vec<f32>> = model.encode(&input).unwrap();

    for i in 0..input.len() {
        for j in 0..input.len() {
            println!(
                "{} ({})\n{} ({})\n{}\n",
                input[i],
                embeddings[i].dot(&embeddings[i]).unwrap(),
                input[j],
                embeddings[j].dot(&embeddings[j]).unwrap(),
                embeddings[i].dot(&embeddings[j]).unwrap()
            );
        }
    }
}