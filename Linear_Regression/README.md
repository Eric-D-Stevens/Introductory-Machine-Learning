# Simple Linear Regression 

This project is a very simple demonstration  of the application of linear regression 
to real estate data in order to predict real estate value. The method is applied to 
the Carnegie Mellon Boston housing dataset. The data and the infomation about the
features can be found [here]( http://lib.stat.cmu.edu/datasets/boston).

## Usage

This project was built using Python 2.7.14. The only libraries that we need access  
to to run this script are the NumPy library and MatPlotLib. The data is included in 
the repository. This data could be replaced by another dataset with the same number 
of features and run properly. 

### Prerequisites

To run this script you will need Python 2.7 and NumPy
 system or using it for a little demo

## Running, Input, and Output 

### Running

To run this file, get into the 'Linear_Regression' directory and enter

```
Python2.7 HousingPriceRegression.py
```
### Input

The name of the input files are `housing_train.txt` and `housing_test.txt` for the
training and testing data respectivly. These files can be replaced assuming the 
replacement files are formated the same way.

### Output

The first output to the terminal is a demonstration of the results of adding 
a dummy variable of all ones.

```
Training Data With Dummy SSE:  9561.19128998
Test Data With Dummy SSE:  1675.23096595
Training Data Without Dummy SSE:  12467.4299083
Test Data Without Dummy SSE:  1589.74933546
```
As we can see, the addition of the dummy variable increases the accuracy of the
training data but actually causes overfitting and therefore decreases the ability
of the model to predict the outcomes of the test set. 

<br />
<br />

Now we will demonstrate the result of adding random variables 

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

