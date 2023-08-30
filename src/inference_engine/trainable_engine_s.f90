! Copyright (c), The Regents of the University of California
! Terms of use are as specified in LICENSE.txt
submodule(trainable_engine_m) trainable_engine_s
  use assert_m, only : assert
  use intrinsic_array_m, only : intrinsic_array_t
  use input_output_pair_m, only : input_output_pair_t
  use sigmoid_m, only : sigmoid_t
  use tensor_m, only : tensor_t
  implicit none

  integer, parameter :: input_layer = 0

contains

  module procedure num_inputs
    n_in = self%n(input_layer)
  end procedure

  module procedure num_layers
    n_layers = size(self%n,1)
  end procedure

  module procedure assert_consistent

    associate( &
      fully_allocated=>[allocated(self%w),allocated(self%b),allocated(self%n),allocated(self%differentiable_activation_strategy_)] &
    )
      call assert(all(fully_allocated),"trainable_engine_s(assert_consistent): fully_allocated",intrinsic_array_t(fully_allocated))
    end associate

    associate(max_width => maxval(self%n), component_dims => [size(self%b,1), size(self%w,1), size(self%w,2)])
      call assert(all(component_dims == max_width), "trainable_engine_s(assert_consistent): conformable arrays", &
        intrinsic_array_t([max_width,component_dims]))
    end associate

    call assert(lbound(self%n,1)==input_layer, "trainable_engine_s(assert_consistent): n base subsscript", lbound(self%n,1))

  end procedure

  module procedure infer

    real(rkind), allocatable :: z(:,:), a(:,:)
    integer l

    call self%assert_consistent

    associate(w => self%w, b => self%b, n => self%n, output_layer => ubound(self%n,1))

      allocate(z, mold=b)
      allocate(a(maxval(n), input_layer:output_layer)) ! Activations

      a(1:n(input_layer),input_layer) = inputs%values()

      feed_forward: &
      do l = 1,output_layer
        z(1:n(l),l) = matmul(w(1:n(l),1:n(l-1),l), a(1:n(l-1),l-1)) + b(1:n(l),l)
        a(1:n(l),l) = self%differentiable_activation_strategy_%activation(z(1:n(l),l))
      end do feed_forward
 
      outputs = tensor_t(a(1:n(output_layer),output_layer))

    end associate

  end procedure

  module procedure train
    integer i, j, k, l, batch, iter, mini_batch_size, pair
    real(rkind), parameter :: eta = 1.5e0 ! Learning parameter
    real(rkind), allocatable :: z(:,:), a(:,:), y(:), delta(:,:), dcdw(:,:,:), dcdb(:,:)
    real(rkind) cost
    type(tensor_t), allocatable :: inputs(:)
    type(tensor_t), allocatable :: expected_outputs(:)

    ! Adam paramters declaration
    integer nhidden,nodes_max
    real(rkind) r,alpha,ir,rr
    real(rkind) beta1, beta2, epsilon
    real(rkind) obeta1, obeta2
    real(rkind), allocatable :: vdw(:,:,:),sdw(:,:,:), vdwc(:,:,:),sdwc(:,:,:)
    real(rkind), allocatable :: vdb(:,:),sdb(:,:), vdbc(:,:), sdbc(:,:)

    ! ! export cost array
    integer export_length, p, unit_num
    real(rkind), allocatable :: export_cost(:)

    export_length = size(mini_batches)

    
    ! integer p, unit_num

    ! ! Fill the array with some values
    export_cost = [(0.0, p = 1,export_length)]


    ! Debug message
    ! character(100) :: message

    ! Network parameters
    nhidden = self%num_layers()-2 ! exclude input and output layers
    nodes_max = maxval(self%n) ! find max neuron num in layers

    ! Adam parameters initialization
    beta1 = .9
    beta2 = .999
    obeta1 = 1.d0 - beta1
    obeta2 = 1.d0 - beta2
    epsilon = 1.d-8

    ! Training parameters initialization
    ! alpha = 1.5d0 ! Learning parameter, replaced by 'eta'

    ! Training parameters initialization
    allocate(vdw(nodes_max,nodes_max,nhidden+1)) 
    allocate(sdw(nodes_max,nodes_max,nhidden+1))
    allocate(vdb(nodes_max,nhidden+1))
    allocate(sdb(nodes_max,nhidden+1)) 

    allocate(vdwc(nodes_max,nodes_max,nhidden+1)) 
    allocate(sdwc(nodes_max,nodes_max,nhidden+1))
    allocate(vdbc(nodes_max,nhidden+1))
    allocate(sdbc(nodes_max,nhidden+1)) 

    vdw = 0.d0
    sdw = 1.d0
    vdb = 0.d0
    sdb = 1.d0

   
    
    


    call self%assert_consistent

    associate(output_layer => ubound(self%n,1))
      
      allocate(a(maxval(self%n), input_layer:output_layer)) ! Activations
      allocate(dcdw,  mold=self%w) ! Gradient of cost function with respect to weights
      allocate(z,     mold=self%b) ! z-values: Sum z_j^l = w_jk^{l} a_k^{l-1} + b_j^l
      allocate(delta, mold=self%b)
      allocate(dcdb,  mold=self%b) ! Gradient of cost function with respect with biases

      associate(w => self%w, b => self%b, n => self%n)
        print *,"Mini Batch Shape in training: ",shape(mini_batches)
        ! Construct a message for debugging
        ! write(message, '(A,F6.2)') "Mini Batch Shape in training: ", shape(mini_batches)
        ! Print the message to stdout
        ! write(*, *) message

        iterate_across_batches: &
        do iter = 1, size(mini_batches)

          cost = 0.; dcdw = 0.; dcdb = 0.
          
          associate(input_output_pairs => mini_batches(iter)%input_output_pairs())
            inputs = input_output_pairs%inputs()
            expected_outputs = input_output_pairs%expected_outputs()
            mini_batch_size = size(input_output_pairs)
          end associate

          iterate_through_batch: &
          do pair = 1, mini_batch_size

            a(1:self%num_inputs(), input_layer) = inputs(pair)%values()
            y = expected_outputs(pair)%values()

            feed_forward: &
            do l = 1,output_layer
              z(1:n(l),l) = matmul(w(1:n(l),1:n(l-1),l), a(1:n(l-1),l-1)) + b(1:n(l),l)
              a(1:n(l),l) = self%differentiable_activation_strategy_%activation(z(1:n(l),l))
            end do feed_forward

            ! cost = cost + sum((y(1:n(output_layer))-a(1:n(output_layer),output_layer))**2)/(2.e0*mini_batch_size)
            cost = sum((y(1:n(output_layer))-a(1:n(output_layer),output_layer))**2)/(2.e0*mini_batch_size)
            export_cost(iter) = cost



            delta(1:n(output_layer),output_layer) = &
              (a(1:n(output_layer),output_layer) - y(1:n(output_layer))) &
              * self%differentiable_activation_strategy_%activation_derivative(z(1:n(output_layer),output_layer))
            
            associate(n_hidden => self%num_layers()-2)
              back_propagate_error: &
              do l = n_hidden,1,-1
                delta(1:n(l),l) = matmul(transpose(w(1:n(l+1),1:n(l),l+1)), delta(1:n(l+1),l+1)) &
                  * self%differentiable_activation_strategy_%activation_derivative(z(1:n(l),l))
              end do back_propagate_error
            end associate

            sum_gradients: &
            do l = 1,output_layer
              dcdb(1:n(l),l) = dcdb(1:n(l),l) + delta(1:n(l),l)
              do concurrent(j = 1:n(l))
                dcdw(j,1:n(l-1),l) = dcdw(j,1:n(l-1),l) + a(1:n(l-1),l-1)*delta(j,l)
              end do
            end do sum_gradients
    
          end do iterate_through_batch
        
          ! adjust_weights_and_biases: &
          ! do l = 1,output_layer
          !   dcdb(1:n(l),l) = dcdb(1:n(l),l)/mini_batch_size
          !   b(1:n(l),l) = b(1:n(l),l) - eta*dcdb(1:n(l),l) ! Adjust biases
          !   dcdw(1:n(l),1:n(l-1),l) = dcdw(1:n(l),1:n(l-1),l)/mini_batch_size
          !   w(1:n(l),1:n(l-1),l) = w(1:n(l),1:n(l-1),l) - eta*dcdw(1:n(l),1:n(l-1),l) ! Adjust weights
          ! end do adjust_weights_and_biases

          ! Adam Optimzer
          adjust_weights_and_biases: &
          do l = 1,output_layer
            dcdb(1:n(l),l) = dcdb(1:n(l),l)/real(mini_batch_size)
            vdb(1:n(l),l) = beta1*vdb(1:(l),l) + obeta1*dcdb(1:n(l),l)
            sdb(1:n(l),l) = beta2*sdb(1:n(l),l) + obeta2*(dcdb(1:n(l),l)**2)
            vdbc(1:n(l),l) = vdb(1:n(l),l)/(1.d0 - beta1**iter) 
            sdbc (1:n(l),l) = sdb(1:n(l),l)/(1.d0 - beta2**iter) 
            b(1:n(l),l) = b(1:n(l),l) - eta*vdbc(1:n(l),l)/(sqrt(sdbc(1:n(l),l))+epsilon) ! Adjust biases using Adam optimization          
            ! --------------------------------------------------------------------------------

            dcdw(1:n(l),1:n(l-1),l) = dcdw(1:n(l),1:n(l-1),l)/real(mini_batch_size)
            vdw(1:n(l),1:n(l-1),l) = beta1*vdw(1:n(l),1:n(l-1),l) + obeta1*dcdw(1:n(l),1:n(l-1),l)
            sdw(1:n(l),1:n(l-1),l) = beta2*sdw(1:n(l),1:n(l-1),l) + obeta2*(dcdw(1:n(l),1:n(l-1),l)**2)
            vdwc(1:n(l),1:n(l-1),l) = vdw(1:n(l),1:n(l-1),l)/(1.d0 - beta1**iter) 
            sdwc(1:n(l),1:n(l-1),l) = sdw(1:n(l),1:n(l-1),l)/(1.d0 - beta2**iter) 
            w(1:n(l),1:n(l-1),l) = w(1:n(l),1:n(l-1),l) - eta*vdwc(1:n(l),1:n(l-1),l)/(sqrt(sdwc(1:n(l),1:n(l-1),l))+epsilon) ! Adjust weights using Adam optimization             
          end do adjust_weights_and_biases


        end do iterate_across_batches

      end associate
    end associate

    ! ! Open file for writing
    unit_num = 10 ! This is an arbitrary number for the unit; choose one that's not being used
    open(unit=unit_num, file='output.txt', status='unknown', action='write')

    ! ! Loop through the data and write to the file
    do i = 1, export_length
      write(unit_num,*) export_cost(i)
    end do

    ! ! Close the file
    close(unit_num)

    
  end procedure

  module procedure construct_from_padded_arrays

     trainable_engine%metadata_ = metadata
     trainable_engine%n = nodes
     trainable_engine%w = weights
     trainable_engine%b = biases
     trainable_engine%differentiable_activation_strategy_ = differentiable_activation_strategy

     call trainable_engine%assert_consistent
  end procedure

  module procedure to_inference_engine
    inference_engine = inference_engine_t(metadata = self%metadata_, weights = self%w, biases = self%b, nodes = self%n)
  end procedure

end submodule trainable_engine_s
