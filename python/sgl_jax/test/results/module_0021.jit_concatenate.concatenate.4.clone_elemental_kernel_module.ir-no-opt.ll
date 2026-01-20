; ModuleID = '__compute_module_concatenate.4.clone_elemental_kernel_module'
source_filename = "__compute_module_concatenate.4.clone_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelThreadDim = type { i64, i64, i64 }
%XLA_CPU_KernelThread = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@concatenate.4.clone_parallel_bounds = private constant [5 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 2000]], [1 x [2 x i64]] [[2 x i64] [i64 2000, i64 4000]], [1 x [2 x i64]] [[2 x i64] [i64 4000, i64 6000]], [1 x [2 x i64]] [[2 x i64] [i64 6000, i64 8000]], [1 x [2 x i64]] [[2 x i64] [i64 8000, i64 10000]]]

; Function Attrs: uwtable
define ptr @concatenate.4.clone_kernel(ptr %0) #0 {
  %concatenate.4.clone.invar_address.dim.1 = alloca i64, align 8
  %concatenate.4.clone.invar_address.dim.0 = alloca i64, align 8
  %tdims_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0
  %tdims = load ptr, ptr %tdims_gep, align 8
  %tdim_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 0
  %tdim_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 1
  %tdim_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 2
  %tdim_x = load i64, ptr %tdim_x_gep, align 4
  %tdim_y = load i64, ptr %tdim_y_gep, align 4
  %tdim_z = load i64, ptr %tdim_z_gep, align 4
  %tid_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %tids = load ptr, ptr %tid_gep, align 8
  %tid_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 0
  %tid_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 1
  %tid_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 2
  %tid_x = load i64, ptr %tid_x_gep, align 4
  %tid_y = load i64, ptr %tid_y_gep, align 4
  %tid_z = load i64, ptr %tid_z_gep, align 4
  %args_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args = load ptr, ptr %args_gep, align 8
  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0
  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %args_gep1 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args2 = load ptr, ptr %args_gep1, align 8
  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %args_gep3 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args4 = load ptr, ptr %args_gep3, align 8
  %arg2_gep = getelementptr %XLA_CPU_KernelArg, ptr %args4, i32 2, i32 0
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %args_gep5 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args6 = load ptr, ptr %args_gep5, align 8
  %arg3_gep = getelementptr %XLA_CPU_KernelArg, ptr %args6, i32 3, i32 0
  %arg3 = load ptr, ptr %arg3_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !3
  %lo_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @concatenate.4.clone_parallel_bounds, i32 0, i64 %tid_x, i32 0, i32 0
  %up_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @concatenate.4.clone_parallel_bounds, i32 0, i64 %tid_x, i32 0, i32 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 4
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 4
  store i64 %lo_dim_0, ptr %concatenate.4.clone.invar_address.dim.0, align 4
  br label %concatenate.4.clone.loop_header.dim.0

concatenate.4.clone.loop_header.dim.0:            ; preds = %concatenate.4.clone.loop_exit.dim.1, %1
  %concatenate.4.clone.indvar.dim.0 = load i64, ptr %concatenate.4.clone.invar_address.dim.0, align 4
  %2 = icmp uge i64 %concatenate.4.clone.indvar.dim.0, %up_dim_0
  br i1 %2, label %concatenate.4.clone.loop_exit.dim.0, label %concatenate.4.clone.loop_body.dim.0

concatenate.4.clone.loop_body.dim.0:              ; preds = %concatenate.4.clone.loop_header.dim.0
  store i64 0, ptr %concatenate.4.clone.invar_address.dim.1, align 4
  br label %concatenate.4.clone.loop_header.dim.1

concatenate.4.clone.loop_header.dim.1:            ; preds = %concatenate.4.clone.merge, %concatenate.4.clone.loop_body.dim.0
  %concatenate.4.clone.indvar.dim.1 = load i64, ptr %concatenate.4.clone.invar_address.dim.1, align 4
  %3 = icmp uge i64 %concatenate.4.clone.indvar.dim.1, 64
  br i1 %3, label %concatenate.4.clone.loop_exit.dim.1, label %concatenate.4.clone.loop_body.dim.1

concatenate.4.clone.loop_body.dim.1:              ; preds = %concatenate.4.clone.loop_header.dim.1
  br label %concatenate.pivot.16.

concat_index_from_operand_id0:                    ; preds = %concatenate.pivot.0.
  %4 = phi i64 [ 0, %concatenate.pivot.0. ]
  %5 = sub nsw i64 %concatenate.4.clone.indvar.dim.1, %4
  %6 = getelementptr inbounds [10000 x [16 x float]], ptr %arg0, i64 0, i64 %concatenate.4.clone.indvar.dim.0, i64 %5
  %7 = load float, ptr %6, align 4, !invariant.load !1, !noalias !6
  br label %concatenate.4.clone.merge

concat_index_from_operand_id1:                    ; preds = %concatenate.pivot.16.8
  %8 = phi i64 [ 16, %concatenate.pivot.16.8 ]
  %9 = sub nsw i64 %concatenate.4.clone.indvar.dim.1, %8
  %10 = getelementptr inbounds [10000 x [24 x float]], ptr %arg1, i64 0, i64 %concatenate.4.clone.indvar.dim.0, i64 %9
  %11 = load float, ptr %10, align 4, !invariant.load !1, !noalias !6
  br label %concatenate.4.clone.merge

concat_index_from_operand_id2:                    ; preds = %concatenate.pivot.40.9
  %12 = phi i64 [ 40, %concatenate.pivot.40.9 ]
  %13 = sub nsw i64 %concatenate.4.clone.indvar.dim.1, %12
  %14 = getelementptr inbounds [10000 x [24 x float]], ptr %arg2, i64 0, i64 %concatenate.4.clone.indvar.dim.0, i64 %13
  %15 = load float, ptr %14, align 4, !invariant.load !1, !noalias !6
  br label %concatenate.4.clone.merge

concatenate.pivot.16.:                            ; preds = %concatenate.4.clone.loop_body.dim.1
  %16 = icmp ult i64 %concatenate.4.clone.indvar.dim.1, 16
  br i1 %16, label %concatenate.pivot.0., label %concatenate.pivot.40.

concatenate.pivot.0.:                             ; preds = %concatenate.pivot.16.
  br label %concat_index_from_operand_id0

concatenate.pivot.40.:                            ; preds = %concatenate.pivot.16.
  %17 = icmp ult i64 %concatenate.4.clone.indvar.dim.1, 40
  br i1 %17, label %concatenate.pivot.16.8, label %concatenate.pivot.40.9

concatenate.pivot.16.8:                           ; preds = %concatenate.pivot.40.
  br label %concat_index_from_operand_id1

concatenate.pivot.40.9:                           ; preds = %concatenate.pivot.40.
  br label %concat_index_from_operand_id2

concatenate.4.clone.merge:                        ; preds = %concat_index_from_operand_id2, %concat_index_from_operand_id1, %concat_index_from_operand_id0
  %18 = phi float [ %7, %concat_index_from_operand_id0 ], [ %11, %concat_index_from_operand_id1 ], [ %15, %concat_index_from_operand_id2 ]
  %19 = getelementptr inbounds [10000 x [64 x float]], ptr %arg3, i64 0, i64 %concatenate.4.clone.indvar.dim.0, i64 %concatenate.4.clone.indvar.dim.1
  store float %18, ptr %19, align 4, !alias.scope !6
  %invar.inc7 = add nuw nsw i64 %concatenate.4.clone.indvar.dim.1, 1
  store i64 %invar.inc7, ptr %concatenate.4.clone.invar_address.dim.1, align 4
  br label %concatenate.4.clone.loop_header.dim.1

concatenate.4.clone.loop_exit.dim.1:              ; preds = %concatenate.4.clone.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %concatenate.4.clone.indvar.dim.0, 1
  store i64 %invar.inc, ptr %concatenate.4.clone.invar_address.dim.0, align 4
  br label %concatenate.4.clone.loop_header.dim.0, !llvm.loop !9

concatenate.4.clone.loop_exit.dim.0:              ; preds = %concatenate.4.clone.loop_header.dim.0
  br label %return

return:                                           ; preds = %concatenate.4.clone.loop_exit.dim.0
  ret ptr null
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 0}
!1 = !{}
!2 = !{i64 640000}
!3 = !{i64 64}
!4 = !{i64 960000}
!5 = !{i64 2560000}
!6 = !{!7}
!7 = !{!"result slice: {index:0, offset:0, size:2560000}", !8}
!8 = !{!"XLA host kernel concatenate.4.clone_kernel AA domain"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.disable"}
