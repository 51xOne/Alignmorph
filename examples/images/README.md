# Input images and pair index

These 21 pairs contain semantically corresponding cars, motorcycles, and dogs
with large spatial displacements and differences in pose or layout.

The 27 unique images are stored once in this
folder; endpoints reused by multiple pairs have the same filename and contents.
The [full manifest](../pairs21.jsonl) also contains both captions and preserves
pair order. The [quick-start manifest](../pairs.jsonl) selects the three pairs
with bundled correspondence maps: `car_01`, `dog_01`, and `dog_08`.

Pairing and evaluation selection are provided by AlignMorph. Source photographs
are from Morph4Data; see [attribution](../../NOTICE.md).

| Pair | Source image | Target image | Bundled map |
| --- | --- | --- | --- |
| car_01 | [car1_1.png](car1_1.png) | [car2_0.png](car2_0.png) | Yes |
| car_02 | [car1_0.png](car1_0.png) | [car2_1.png](car2_1.png) | Generate |
| car_03 | [car2_0.png](car2_0.png) | [car1_0.png](car1_0.png) | Generate |
| motorcycle_01 | [motorcycle1_1.png](motorcycle1_1.png) | [motorcycle5_0.png](motorcycle5_0.png) | Generate |
| motorcycle_02 | [motorcycle3_0.png](motorcycle3_0.png) | [motorcycle2_1.png](motorcycle2_1.png) | Generate |
| motorcycle_03 | [motorcycle4_1.png](motorcycle4_1.png) | [motorcycle3_1.png](motorcycle3_1.png) | Generate |
| dog_01 | [dog_0.png](dog_0.png) | [dog2_1.png](dog2_1.png) | Yes |
| dog_02 | [dog2_0.png](dog2_0.png) | [dog_1.png](dog_1.png) | Generate |
| dog_03 | [dog1_0.png](dog1_0.png) | [dog2_1.png](dog2_1.png) | Generate |
| dog_04 | [dog6_0.jpg](dog6_0.jpg) | [dog7_0.JPEG](dog7_0.JPEG) | Generate |
| dog_05 | [dog4_0.png](dog4_0.png) | [dog6_1.jpg](dog6_1.jpg) | Generate |
| dog_06 | [dog5_1.jpeg](dog5_1.jpeg) | [dog_1.png](dog_1.png) | Generate |
| car_04 | [car1_0.png](car1_0.png) | [car1_1.png](car1_1.png) | Generate |
| car_05 | [car5_0.jpg](car5_0.jpg) | [car5_1.jpg](car5_1.jpg) | Generate |
| car_06 | [car2_0.png](car2_0.png) | [car2_1.png](car2_1.png) | Generate |
| car_07 | [car3_0.png](car3_0.png) | [car3_1.png](car3_1.png) | Generate |
| dog_07 | [dog_0.png](dog_0.png) | [dog_1.png](dog_1.png) | Generate |
| dog_08 | [dog2_0.png](dog2_0.png) | [dog2_1.png](dog2_1.png) | Yes |
| dog_09 | [dog3_0.jpg](dog3_0.jpg) | [dog3_1.jpg](dog3_1.jpg) | Generate |
| dog_10 | [dog6_0.jpg](dog6_0.jpg) | [dog6_1.jpg](dog6_1.jpg) | Generate |
| dog_11 | [dog7_0.JPEG](dog7_0.JPEG) | [dog7_1.JPEG](dog7_1.JPEG) | Generate |
