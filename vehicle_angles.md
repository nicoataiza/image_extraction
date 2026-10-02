# Exterior

## Entire vehicle
- id: `vehicle_exterior`
- description: Full exterior view of the entire vehicle.
- framing: The entire vehicle in frame from roof to tyres, with the full front or rear bumper and the ground beneath it visible and margin around the car.
- avoid: A close-up of part of the vehicle, cropped so the roof, bumper or wheels are cut off.
- views:
  - front
  - front-left
  - left-side
  - rear-left
  - rear
  - rear-right
  - right-side
  - front-right
- capture: wide
- aliases:
  - whole vehicle
  - vehicle exterior
  - 360 vehicle view

## Headlights
- id: `headlight`
- description: Close-up view of the vehicle's front headlight assembly.
- views:
  - front-left
  - front-right
- capture: close-up
- aliases:
  - headlamp
  - front light

## Taillights
- id: `taillight`
- description: Close-up view of the vehicle's rear taillight assembly.
- views:
  - rear-left
  - rear-right
- capture: close-up
- aliases:
  - tail light
  - rear light

## Wheels and tyres
- id: `wheel_tyre`
- description: Wheel and tyre clearly visible, including rim and tyre condition.
- views:
  - front-left
  - front-right
  - rear-left
  - rear-right
- capture: close-up
- aliases:
  - wheel
  - tyre
  - tire
  - rim

## Body panels
- id: `body_panel`
- description: Individual exterior body panel clearly visible.
- parts:
  - bonnet
  - front-left-door
  - front-right-door
  - rear-left-door
  - rear-right-door
  - front-left-guard
  - front-right-guard
  - front-bumper
  - rear-bumper
  - left-quarter-panel
  - right-quarter-panel
- capture: medium

## Door mirrors
- id: `door_mirror`
- description: Exterior door mirror assembly.
- sides:
  - left
  - right
- views:
  - front
  - rear
- aliases:
  - side mirror
  - wing mirror

## Towbar
- id: `towbar`
- description: Rear towbar and mounting area clearly visible.
- capture: close-up

## Roof accessories
- id: `roof_accessories`
- parts:
  - roof-rails
  - roof-rack
  - antenna
  - sunroof

## Snorkel
- id: `snorkel`
- description: Exterior engine air-intake snorkel where fitted.
- optional: true


# Identification

## VIN
- id: `vin`
- description: VIN or vehicle identification plate with identifying text readable.
- framing: The complete identification plate or label in frame, flat-on and sharp, with all of its printed text readable.
- avoid: A blurry, angled or partly cropped plate whose text cannot be read.
- capture: close-up
- requirement: Capture all vehicle identification plates or visible VIN markings present in the car.
- aliases:
  - VIN plate
  - vehicle identification number
  - compliance plate


# Engine Bay

## Engine bay overview
- id: `engine_bay`
- description: Clean wide view of the complete engine bay.
- framing: A wide view of the entire open engine bay from the front, with the whole engine and both inner guards visible.
- avoid: A close-up or angled view of only part of the engine, cropped at the edges.
- capture: wide
- views:
  - front
  - left-angle
  - right-angle

## Airbox
- id: `engine_airbox`
- description: Engine air filter housing / airbox.
- capture: close-up
- aliases:
  - air box
  - air filter box

## Brake booster
- id: `brake_booster`
- description: Brake booster assembly in the engine bay.
- capture: close-up
- aliases:
  - booster

## ABS pump
- id: `abs_pump`
- description: ABS hydraulic pump/module in the engine bay.
- capture: close-up
- aliases:
  - ABS module

## Fuel filter housing
- id: `fuel_filter_housing`
- description: Fuel filter housing in the engine bay.
- capture: close-up
- aliases:
  - fuel filter

## Battery
- id: `battery`
- description: Vehicle battery and surrounding mounting area.
- capture: close-up

## Fusebox
- id: `engine_fusebox`
- description: Fuse or relay box located in the engine bay.
- capture: close-up
- aliases:
  - fuse box
  - relay box

## Engine cold side
- id: `engine_cold_side`
- description: Cold-side region of the engine and associated components.
- capture: medium

## Engine hot side
- id: `engine_hot_side`
- description: Hot-side region of the engine and associated components.
- capture: medium


# Interior — Overview

## Dashboard
- id: `dashboard`
- description: Wide view of the dashboard and front cabin.
- capture: wide
- views:
  - driver-side
  - centre
  - passenger-side
- aliases:
  - dash

## Instrument cluster
- id: `instrument_cluster`
- description: Driver instrument cluster with speedometer and gauges visible.
- capture: close-up
- aliases:
  - speedometer cluster
  - gauge cluster
  - dash cluster

## Front seats and keys
- id: `front_seats`
- description: Front driver and passenger seats, with vehicle keys visible where required.
- capture: wide

## Rear seats
- id: `rear_seats`
- description: Rear seating area.
- views:
  - rear-left
  - rear-right
- capture: wide

## Roof lining
- id: `roof_lining`
- description: Interior headlining / roof lining.
- capture: medium
- aliases:
  - headliner

## Interior mirror
- id: `interior_mirror`
- description: Interior rear-view mirror.
- capture: close-up
- aliases:
  - rear-view mirror
  - rearview mirror

## Sunvisors
- id: `sunvisor`
- sides:
  - driver
  - passenger
- capture: close-up

## Pedals
- id: `pedals`
- description: Driver footwell showing accelerator, brake and clutch where fitted.
- capture: close-up


# Interior — Controls and Trim

## Steering wheel
- id: `steering_wheel`
- description: Steering wheel and steering-wheel controls.
- capture: medium

## Combination switches
- id: `combination_switch`
- description: Stalk switches behind the steering wheel.
- sides:
  - left
  - right
- aliases:
  - indicator stalk
  - wiper stalk
  - steering column switch

## Power window switches
- id: `power_window_switch`
- description: Power-window switch panel.
- requirement: Capture all fitted switch panels.
- capture: close-up

## Switch panels
- id: `switch_panel`
- description: Interior control or switch panel.
- requirement: Capture all fitted switch panels.
- capture: close-up

## Door trims
- id: `door_trim`
- description: Interior door trim/card.
- sides:
  - front-left
  - front-right
  - rear-left
  - rear-right
- aliases:
  - door card
  - door panel

## Centre console
- id: `centre_console`
- description: Centre console between the front seats.
- capture: medium
- aliases:
  - center console

## Console lid
- id: `console_lid`
- description: Centre-console storage lid/armrest.
- capture: close-up

## Glovebox
- id: `glovebox`
- description: Passenger-side glovebox.
- capture: medium
- aliases:
  - glove box

## Head unit
- id: `head_unit`
- description: Dashboard infotainment or audio head unit.
- capture: close-up
- aliases:
  - stereo
  - infotainment screen
  - radio

## Heater controls
- id: `heater_controls`
- description: Front HVAC/heater control panel.
- capture: close-up
- aliases:
  - climate controls
  - HVAC controls

## Rear heater controls
- id: `rear_heater_controls`
- description: Rear passenger HVAC/heater control panel, where fitted.
- capture: close-up
- optional: true

## Gear shifter
- id: `gear_shifter`
- description: Gear selector/shifter and surrounding trim.
- capture: close-up
- aliases:
  - gear lever
  - gear selector
