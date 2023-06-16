FILE=$1

if [ $FILE == 'CelebA' ]
then
    URL=https://www.dropbox.com/s/3e5cmqgplchz85o/CelebA_nocrop.zip?dl=0
    ZIP_FILE=./data_celeba/CelebA.zip
else
    echo "Available datasets are: CelebA"
    exit 1
fi

mkdir -p ./data_celeba/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./data_celeba/

if [ $FILE == 'CelebA' ]
then
    mv ./data_celeba/CelebA_nocrop ./data_celeba/CelebA
fi

rm $ZIP_FILE
