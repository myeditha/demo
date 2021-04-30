import csv
import sys
import random

if __name__ == "__main__":
    csv.field_size_limit(sys.maxsize)
    random.seed(42)
    split = 0.7
    with open('archive/stories.csv') as csvfile:
        spamreader = csv.reader(csvfile)
        with open('archive/stories.test.csv', "w") as test:
            testwriter = csv.writer(test)
            with open('archive/stories.train.csv', "w") as train:
                trainwriter = csv.writer(train)
                trainwriter.writeheader()
                testwriter.writeheader()
                for row in spamreader:
                    randnum = random.uniform(0,1)
                    if(randnum > split):
                        testwriter.writerow(row)
                    else:
                        trainwriter.writerow(row)

