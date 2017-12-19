// https://www.kaggle.com/uciml/sms-spam-collection-dataset/data

import fs from 'fs';
import csv from 'fast-csv';
import synchronizedShuffle from 'synchronized-array-shuffle';
import sanitizeHtml from 'sanitize-html';
import urlRegex from 'url-regex';
import natural from 'natural';
import svm from 'svm';

natural.PorterStemmer.attach();

const SVM = new svm.SVM();

const M = 5572;
const MTest = 5000;
const MTrain = M - MTest;

let X = [];
let y = [];

const stream = fs
  .createReadStream('./src/spam.csv')
  .pipe(csv())
  .on('data', fromFile)
  .on('end', init);

function fromFile(data) {
  if (data[0] !== 'spam' && data[0] !== 'ham') {
    return;
  }

  let label = data[0];
  let feature = data[1];

  y.push(label);
  X.push(feature);
}

function init() {
  /* ----------------------------- */
  //  Train & Predict with Natural //
  /* ----------------------------- */

  // console.log('Started training with Natural, this will take time ...');

  // const shuffled = synchronizedShuffle([ X, y ]);

  // const { XTrain, yTrain, XTest, yTest } = getDataSets(shuffled[0], shuffled[1], MTrain);
  // const classifier = trainWithNaturalsBayesClassifier(XTrain, yTrain);

  // const YPredict = XTest.map((v) => classifier.classify(v));
  // const trueHits = YPredict.filter((v, i) => v === yTest[i]).length;
  // const accuracy = (trueHits / MTest) * 100;
  // console.log(accuracy);

  // return;

  /* -------------------- */
  //  Preparation for SVM //
  /* -------------------- */

  // Lower-casing:
  // The entire SMS is converted into lower case, so that captialization is ignored (e.g., DoLLar is treated the same as Dollar).
  X = X.map(string => string.toLowerCase());

  // Stripping HTML:
  // All HTML tags are removed from content.
  // It may be not effective for in this scenario of a SMS spam filter, but in other scenarios (e.g. emails).
  // We remove all the HTML tags, so that only the content remains.
  X = X.map(sanitizeHtml);

  // Normalizing URLs:
  // All URLs are replaced with the text "httpaddress".
  X = X.map(normalizeUrl);
  // console.log(X.filter(string => string.includes('httpaddress')));

  // Normalizing Email Addresses:
  // All email addresses are replaced with the text "emailaddress".
  X = X.map(normalizeEmailAddresses);
  // console.log(X.filter(string => string.includes('emailaddress')));

  // Normalizing Numbers:
  // All numbers are replaced with the text "number".
  X = X.map(normalizeNumbers);
  // console.log(X.filter(string => string.includes('number')));

  // Normalizing Dollars:
  // All dollar signs ($) are replaced with the text "dollar".
  X = X.map(normalizeDollarSigns);
  // console.log(X.filter(string => string.includes('dollar')));

  // Removal of non-words:
  // Non-words and punctuation have been removed.
  X = X.map(removeNonAlphanumericChars);
  // console.log(X.filter(string => string.includes('dollar')));

  // Word Stemming and Tokenizing:
  // Words are reduced to their stemmed form.
  // For example, "discount”, "discounts”, "discounted” and "discounting” are all replaced with "discount”.
  // Strings are tokenized into an array of words.
  // console.log(X[2]);
  X = X.map(string => string.tokenizeAndStem());
  // console.log(X[2]);

  const vocabularyList = createVocabularyList(X);
  X = X.map(featureExtraction(vocabularyList));

  /* ------------------------ */
  //  Train & Predit with SVM //
  /* ------------------------ */

  console.log('Started training with SVM, this will take time ...');

  // Transform to expected format for target vector by SVM library
  y = y.map(v => v === 'spam' ? 1 : -1);

  const shuffled = synchronizedShuffle([ X, y ]);

  const { XTrain, yTrain, XTest, yTest } = getDataSets(shuffled[0], shuffled[1], MTrain);

  SVM.train(XTrain, yTrain);

  const YPredict = SVM.predict(XTest);
  const trueHits = YPredict.filter((v, i) => v === yTest[i]).length;
  const accuracy = (trueHits / MTest) * 100;

  console.log(accuracy);
}

function normalizeUrl(string) {
  const matchedUrls = string.match(urlRegex());

  (matchedUrls || []).forEach(url => {
    const regEx = new RegExp(url, 'g');
    string = string.replace(regEx, ' httpaddress ');

    // Alternative
    // string = string.split(url).join('httpaddress');
  });

  return string;
}

function normalizeEmailAddresses(string) {
  // Simple regular expression for email extraction
  // Which is good, because we don't want to have exact email macthes
  // But all kind of look alike email matches
  const regEx = new RegExp(/\S+[a-z0-9]@[a-z0-9\.]+/img);
  return string.replace(regEx, ' emailaddress ');
}

function normalizeNumbers(string) {
  const regEx = new RegExp(/\d+/, 'g');
  return string.replace(regEx, ' number ');
}

function normalizeDollarSigns(string) {
  const regEx = new RegExp(/\$/, 'g');
  return string.replace(regEx, ' dollar ');
}

function removeNonAlphanumericChars(string) {
  const regEx = new RegExp(/[^0-9a-z]/, 'gi');
  return string.replace(regEx, ' ');
}

function createVocabularyList(X) {
  const groupedByWord = X.reduce((group, xs) => {
    xs.forEach(word => {
      if (group[word]) {
        group[word]++;
      } else {
        group[word] = 1;
      }
    });

    return group;
  }, {});

  const POPULARITY = 5;
  const groupedByPopularWord = Object.keys(groupedByWord).reduce((list, word) => {
    const count = groupedByWord[word];

    if (count > POPULARITY) {
      list.push(word);
    }

    return list;
  }, []);
  // console.log(`${groupedByPopularWord.length} words in vocabulary list`);
  // console.log('adjust popularity for more or less words in list');

  return groupedByPopularWord;
}

function featureExtraction(vocabularyList) {
  return function createFeatureVector(xs) {
    return vocabularyList.map((vocable) => {
      if (xs.includes(vocable)) {
        return 1;
      } else {
        return 0;
      }
    });
  }
}

// Training Set / Test Set

function getDataSets(X, y, size) {
  if (size > M) {
    return;
  }

  const XTrain = [];
  const yTrain = [];
  const XTest = [];
  const yTest = [];

  Array(M).fill().map((v, i) => {
    if (i < size) {
      XTrain.push(X[i]);
      yTrain.push(y[i]);
    } else {
      XTest.push(X[i]);
      yTest.push(y[i]);
    }
  });

  return { XTrain, yTrain, XTest, yTest };
}

// Training Phase

function trainWithNaturalsBayesClassifier(XTrain, yTrain) {
  const classifier = new natural.BayesClassifier();

  Array(MTrain).fill().forEach((v, i) => classifier.addDocument(XTrain[i], yTrain[i]));
  classifier.train();

  return classifier;
}

function trainWithSvm(XTrain, yTrain) {
  const classifier = null;

  return classifier;
}