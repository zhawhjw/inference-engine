! Copyright (c), The Regents of the University of California
! Terms of use are as specified in LICENSE.txt
submodule(concurrent_dot_products_m) concurrent_dot_products_s
  use assert_m, only : assert
  use activation_strategy_m, only : activation_strategy_t
  use step_m, only : step_t
  implicit none

contains

  module procedure infer
    
    integer n, layer, m
    integer, parameter :: input_layer = 1
    real(rkind), allocatable :: neuron(:,:), output_values(:), pre_activation_in(:,:), pre_activation_out(:)

    associate(neurons_per_layer => size(input_weights,1), num_layers => size(biases,2))

      allocate(neuron(neurons_per_layer, num_layers))
      allocate(pre_activation_in, mold=neuron)

      do concurrent(n = 1:neurons_per_layer)
        pre_activation_in(n,input_layer) = dot_product(input_weights(n,:), input(:)) + biases(n,input_layer)
        neuron(n,input_layer) = activation_strategy%activation(pre_activation_in(n,input_layer))
      end do
      do layer = 2, num_layers
        if (skip) then
          do concurrent(n = 1:neurons_per_layer)
            pre_activation_in(n,layer) = dot_product(hidden_weights(n,:,layer-1), neuron(:,layer-1)) + biases(n,layer)
            neuron(n,layer) = neuron(n,layer-1) + activation_strategy%activation(pre_activation_in(n,layer))
          end do
        else
          do concurrent(n = 1:neurons_per_layer)
            pre_activation_in(n,layer) = dot_product(hidden_weights(n,:,layer-1), neuron(:,layer-1)) + biases(n,layer)
            neuron(n,layer) = activation_strategy%activation(pre_activation_in(n,layer))
          end do
        end if
      end do

      associate(num_outputs => size(output_weights,1))
        allocate(output_values(num_outputs), pre_activation_out(num_outputs))
        do concurrent(m = 1:num_outputs)
          pre_activation_out(m) = dot_product(output_weights(m,:), neuron(:,num_layers)) + output_biases(m)
          output_values(m) = activation_strategy%activation(pre_activation_out(m))
        end do
        outputs = outputs_t(output_values, pre_activation_in, pre_activation_out)
      end associate
    end associate

  end procedure

end submodule concurrent_dot_products_s
