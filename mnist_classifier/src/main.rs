use mnist_classifier::{evaluate, evaluate_threaded, test_threaded};


fn main() {
    test_threaded();
    evaluate_threaded(true);
    evaluate(true);
}