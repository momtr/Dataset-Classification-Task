let data;
let els = [];
let normalizer;
let train;
let test;

// classifier
let knn;
let bayes;

function preload() {
    data = loadTable('data/classification.csv');
}

function setup() {
    els = DataProcessing.tableDataToEls(data);
    let trainTest = DataProcessing.trainTestSplit(els);
    train = trainTest.train;
    test = trainTest.test;
    // visualize them
    createCanvas(600, 600);
    background(40);
    normalizer = new Normalizer(els);
    for(let i of train) {
        let normalized = normalizer.normalize(i.x);
        let scaled = DataProcessing.scale(normalized, [width, height])
        if(i.y == 0)
            fill(255, 0, 0);
        else
            fill(0, 255, 0);
        ellipse(scaled[0], scaled[1], 5, 5);
    }

    // KNN classifier
    knn = new KNN(train, 3);
    console.log("KNN score: " + knn.score(test) * 100 + "%");

    // Naive Bayes
    let bayesData = {
        classes: [0, 1],
        features: ['age', 'interest'],
        training_data: []
    }
    for(let i of train) {
        let obj = 
        bayesData.training_data.push({
            class: i.y,
            age: i.x[0],
            interest: i.x[1]
        })
    }
    bayes = new NaiveBayesClassifier(bayesData);
    console.log("Bayes score: " + bayes.score(test) * 100 + "%");

}