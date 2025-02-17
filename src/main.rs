use rust_bert::pipelines::sentence_embeddings;

fn main() {
    let model = sentence_embeddings::SentenceEmbeddingsBuilder::remote(sentence_embeddings::SentenceEmbeddingsModelType::AllMiniLmL6V2).create_model()
        .unwrap();

    let input = ["This is a sentence transformer example."];

    let embeddings: Vec<Vec<f32>> = model.encode(&input).unwrap();
}