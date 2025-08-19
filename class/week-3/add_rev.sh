#!/bin/bash

echo "Enter num1 and num2"
read num1 num2

sum=$((num1 + num2))
echo "The sum is: $sum"



echo "Enter a number:"
read num

fac=1

for ((i=2; i<=num; i++))
do
  fac=$((fact * i))
done

echo "Factorial of $num is $fac"