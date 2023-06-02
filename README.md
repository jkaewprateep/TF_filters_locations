# TF_filters_locations
Find location of object by using Tensorflow filters

🧸💬 For tracking location on the screen as we see from many applications or games is very easy but many APIs are released with many features and we could implement them later but we need to understand the concept first. <br>
👧💬 To achevement of our goals by intercepting the ball with acceleration from byside, we need to locate the ball in the current and tracking location for prediction of the target distance and actions to perform. ```actions = { "up": K_w, "none": K_h, "down": K_s }```

### Pre-process image for filters ###
🐑💬 The inputs can be from many target sizes of the player screen, one way is to compact information within scope and we can restore information with lossy/lossless method after testing the input to requirements. The lossy method is the method compress images with target information remaining when lossless is the method to remain most of the information. TF.resize is the lossy method, and an example of the lossless method is lossless encodings or vectors image.
```
image = tf.keras.utils.img_to_array( image, data_format="channels_last" )
image_resized = tf.image.resize( image, size=( 32, 32 ), method=tf.image.ResizeMethod.BILINEAR, 
      preserve_aspect_ratio=False, antialias=False, name="image_resize" )
```

### Remove tuck players - remain ball on the screen ###
🐬🥀💬 We know that the tuck player is the cylindrical shape or rectangular shape we do Normalization and filter out the result is only a ball running on the screen.
```
layer_1 = tf.keras.layers.Normalization(mean=3., variance=2.)( image_resized )
layer_2 = tf.keras.layers.Normalization(mean=4., variance=6.)( layer_1 )
	
image = tf.expand_dims( image_resized, axis=0, name="expand dimension" )
layer_3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')( image )
```

### Reduce dimensions of the image output / channels ###
🧸💬 When we are working with the convolutional layer they provide many benefits of filtering and channels dependency work of an image we can perform individual tasks on specific channels frequency, before we display we need to combine them into an image format.
```
final_layer = tf.keras.layers.Conv2D(1, (2, 1), activation='relu')( layer_3 )
final_layer = tf.squeeze( final_layer, axis=0, name="squeeze" )
image_in_process = final_layer
```

### Target ball location X and Y ###
👧💬 Re-location is easy because we do it since the pre-process, you can consider as cylindrical property in mathematics and convolutional layer but to perform real-image we use masking or scanning method.
```
final_layer = tf.math.argmax( image_in_process, axis=0, output_type=tf.dtypes.int32, name="max_x" )
final_layer = tf.math.argmax( final_layer, axis=0, output_type=tf.dtypes.int32, name="max_x" )

final_layer_2 = tf.math.argmax( image_in_process, axis=1, output_type=tf.dtypes.int32, name="max_y" )
final_layer_2 = tf.math.argmax( final_layer_2, axis=0, output_type=tf.dtypes.int32, name="max_y" )
```

## Sample ball location on screen ##
🐑💬 Now we can have output as X and Y co-ordinates we can create functions to perform actions or predictions in advance more than intercept of the balls but we have more time for the possibility we can create by the later ability we can develop our source to.
![alt text](https://github.com/jkaewprateep/TF_filters_locations/blob/main/Simple%20Filter%20locations.gif)

## Files and Directory ##

File Name | Description |
--- | --- |
Simple Filter locations_0.gif | Image file, accumulation from the ball location |
Simple Filter locations.gif | Image file, 2D location from the ball location |
README.md | Readme file |

## Accumulation of ball direction on screen ##
🐬🥀💬 For considering image input the advantage over time series input is we do not need to modify the input but the model can learn from the actual object and reflection screen. This method can create more accuracy than input coordinates when apply to the right sciences and this method is visualize method where the monitor, rulers, and model are in the same domain.
![alt text](https://github.com/jkaewprateep/TF_filters_locations/blob/main/Simple%20Filter%20locations_0.gif)
