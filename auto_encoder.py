from load_data import *







def main():
    load_train("./ae_data/all_regularized_impurities_train_normal/",
               "./ae_data/all_regularized_impurities_train_anomaly/")
    load_test("./ae_data/all_regularized_impurities_test_normal/",
              "./ae_data/all_regularized_impurities_test_anomaly/")


if __name__ == "__main__":
    main()



