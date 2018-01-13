# Self Organising Wedding Invitations:

I have a lot of photos that I've taken together with my wife over the last few years. When we got married I thought it'd be nice to gather them all up and use machine learning extract some sort of underlying structure from them as a way of discovering something new about our relationship.

I decided to use an algorithm called a Self-Organising Map to arrange our photos into a collage based on colour. I figured that our photos are almost a representitive sample of the way we spend our time together so the colours that become emphasised in the SOM represent the colours of our life together. Turns out that palette is mostly brown and black, but the end result is still pretty nice:

![output](https://user-images.githubusercontent.com/2457362/34902257-23f8551a-f85b-11e7-8e29-6f35c3480d41.png)

## How it works

[Self-Organising Maps](http://davis.wpi.edu/~matt/courses/soms/) are a type of neural network used to map high dimensionality data into usually a one or two dimensional space. In this case, that high dimensionality data are images and I'll map them into a two dimensional collage.

### Feature Extraction

Images are of really high dimensionality so to simplify things I used a feature extraction step to boil them down into just a single representitive colour in [R,G,B] space. To do that I tried:

* clustering a random sample of pixels using k-means then taking the centroid of the largest cluster
* averaging across all pixels

In the end, both approaches produced subjectively the same results so for the sake of simplicity (albeit with a slight increase in computation time) I went with just averaging across pixels.

### Training the SOM

Is pretty standard. I randomly initialised the three weights of each node in the map by sampling from U([0,1]). With the [R,G,B] feature extracted from the image I found the best matching unit as the one having the smallest L1 norm. I then adjusted the weights of the surrounding units towards that of the BMU. The learning rate applied to each unit's weights depended on that unit's euclidean distance from the BMU. The learning process for just one image looks like this:

![som_training](https://user-images.githubusercontent.com/2457362/34902254-1eb35424-f85b-11e7-8b51-8f24b6aaf861.png)

That process is repeated for each image with the end result looking like this (like I said, mostly brown and black with a bit of sky blue thrown in on the bottom left):

![som_trained](https://user-images.githubusercontent.com/2457362/34905430-a86ea4ee-f89b-11e7-9afb-e466c35f78fa.png)

### Laying the photos out on the SOM

With a trained map, I figured I was just about finished; I'd project each photo onto the map and save the resulting image as my nice collage to use on the wedding invitations. Unfortunately, when I plotted the coordinates of each image's BMU on the trained map, there was lots of overlap and some unpopular areas of the map were empty. To get around that I used a greedy algorithm that's more easily described in psuedocode:

```
while the collage isn't completely occupied

    if the list of available photos is empty then
        refill it from the input dataset
    end if

    find the photo in the list of available photos which has the closest match to its BMU
    
    set this photo to 'BMP' (best matching photo)
    
    find the coordinates of the BMP's BMU in the map
    
    find the largest footprint that the BMP can occupy without overlapping with any neighbouring images
    
    set the coordinates of the BMP to the its BMU and mark its footprint on the map as occupied
    
    remove the BMP from the list of available photos
    
end while
```

Because this algorithm gives priority to images that have the closest match to their BMU, those images are able to be placed first which means they can take up a larger area in the map (since they'll have fewer neighbours). As a result, photos which strongly represent their area in the SOM take up more space and the resulting collage has a more coherent grouping of colours. The resulting layout of each image looks like so: 

Here's the representitive colour of each photo plotted at the centroid of its computed footprint:

![layout](https://user-images.githubusercontent.com/2457362/34905433-ae5eb826-f89b-11e7-984c-c25e911fdf19.png)

And this shows the footprint of each image laid out in the collage:

![patchwork](https://user-images.githubusercontent.com/2457362/34902300-22693074-f85c-11e7-8991-e4e8a6353854.png)

To finish up, each photo is rendered to its footprint in the collage and the enormous resulting .png is saved to the working directory.
