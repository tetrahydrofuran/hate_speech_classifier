import logging

from processing import pos_classifier

logging.basicConfig(level=logging.DEBUG)


def main():

    pos_classifier.export_pos_classifier(classifier='bayes', state=42)
    # pos_classifier.export_pos_classifier(classifier='tree', state=42)
    pass

if __name__ == "__main__":
    main()
