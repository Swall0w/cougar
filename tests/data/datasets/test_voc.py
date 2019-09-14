from cougar.data.datasets import get_voc


def test_get_voc():
    dataset = get_voc(root='~/dataset/',
#                      image_set='2007_train',
                      image_set='2007_test',
                      transforms=None
    )
    print(dataset[0])


if __name__ == '__main__':
    test_get_voc()