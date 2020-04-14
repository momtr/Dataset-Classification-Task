// KNN and Naive Bayers Classifiers Test 
// Mitterdorfer, 2020

// data from the file p5.Table
let data;
// all elements
let els = [];
// our normalizer
let normalizer;
// train and test data
let train;
let test;

// classifiers
let knn;
let bayes;

function preload() {
    // load the csv file (dataset)
    data = loadTable('data/classification.csv');
}


function setup() {
    // transform the data into form used by classifiers
    els = DataProcessing.tableDataToEls(data);
    // 25& testing data, 75% training data
    let trainTest = DataProcessing.trainTestSplit(els, 0.75);
    train = trainTest.train;
    test = trainTest.test;
    // visualize datapoints
    createCanvas(600, 600);
    background(40);
    // normalizer (trained on all datapoints)
    normalizer = new Normalizer(els);
    for(let i of train) {
        let normalized = normalizer.normalize(i.x);
        // values between width and height
        let scaled = DataProcessing.scale(normalized, [width, height])
        if(i.y == 0)
            fill(255, 0, 0);
        else
            fill(0, 255, 0);
        // draw the point
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
    // transform data into readable form
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
