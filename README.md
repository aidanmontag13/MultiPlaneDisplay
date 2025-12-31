# MultiPlaneDisplay
This is a repository for code and schematics for multiplane display concept

<p align="center">
  <img src="artifacts/imag1.jpg" width="800">
</p>

# How does it work?
Multi-Plane Display uses image processing and Pepper's Ghost illusions to enable any photo to be viewed with real depth, creating a "2 and a half D" Illusion.
<p align="center">
  <img src="artifacts/graphic.png" width="800">
</p>

## Workflow

1. **Depth segmentation**  
   A depth map is extracted from the image using *Depth Anything v2* and used to segment the scene into **foreground**, **middleground**, and **background** layers.

2. **Viewer tracking**  
   Monocular head tracking estimates the viewer’s position relative to the display.

3. **Shadow correction**  
   Shadow maps are dynamically adjusted based on the viewer’s position to prevent off-axis ghosting artifacts.

4. **Optical display**  
   The layered images are displayed through a parascope mirror array, producing the virtual display stacking effect.
   
<table align="center">
  <tr>
    <td align="center">
      <img src="artifacts/moonlanding2.png" width="400"><br>
      <sub>Displayed Image</sub>
    </td>
    <td width="40"></td>
    <td align="center">
      <img src="artifacts/imag3.jpg" width="425"><br>
      <sub>Viewable Image</sub>
    </td>
  </tr>
</table>




## Results
<p align="center">
  <img src="artifacts/left.jpg" width="300">
  <img src="artifacts/middle.jpg" width="300">
  <img src="artifacts/right.jpg" width="300">
</p>

<p align="center">
  <img src="artifacts/imag2.jpg" width="500">
</p>


