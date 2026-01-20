; ModuleID = '__compute_module_slice.4.clone_elemental_kernel_module'
source_filename = "__compute_module_slice.4.clone_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@slice.4.clone_parallel_bounds = private unnamed_addr constant [4 x [2 x [2 x i64]]] [[2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 0, i64 5000]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 5000, i64 10000]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 3], [2 x i64] [i64 0, i64 5000]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 3], [2 x i64] [i64 5000, i64 10000]]]

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @slice.4.clone_kernel(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %tid_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %tids = load ptr, ptr %tid_gep, align 8
  %tid_x = load i64, ptr %tids, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %lo_dim_0_gep = getelementptr inbounds [4 x [2 x [2 x i64]]], ptr @slice.4.clone_parallel_bounds, i64 0, i64 %tid_x, i64 0, i64 0
  %up_dim_0_gep = getelementptr inbounds [4 x [2 x [2 x i64]]], ptr @slice.4.clone_parallel_bounds, i64 0, i64 %tid_x, i64 0, i64 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 16
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 8
  %lo_dim_1_gep = getelementptr inbounds [4 x [2 x [2 x i64]]], ptr @slice.4.clone_parallel_bounds, i64 0, i64 %tid_x, i64 1, i64 0
  %up_dim_1_gep = getelementptr inbounds [4 x [2 x [2 x i64]]], ptr @slice.4.clone_parallel_bounds, i64 0, i64 %tid_x, i64 1, i64 1
  %lo_dim_1 = load i64, ptr %lo_dim_1_gep, align 16
  %up_dim_1 = load i64, ptr %up_dim_1_gep, align 8
  %.not9 = icmp ult i64 %lo_dim_0, %up_dim_0
  %.not57 = icmp ult i64 %lo_dim_1, %up_dim_1
  %or.cond = select i1 %.not9, i1 %.not57, i1 false
  br i1 %or.cond, label %slice.4.clone.loop_header.dim.1.preheader.us, label %return

slice.4.clone.loop_header.dim.1.preheader.us:     ; preds = %1, %slice.4.clone.loop_header.dim.1.slice.4.clone.loop_exit.dim.1_crit_edge.us
  %slice.4.clone.invar_address.dim.0.010.us = phi i64 [ %invar.inc.us, %slice.4.clone.loop_header.dim.1.slice.4.clone.loop_exit.dim.1_crit_edge.us ], [ %lo_dim_0, %1 ]
  br label %slice.4.clone.loop_header.dim.2.preheader.us

slice.4.clone.loop_header.dim.2.preheader.us:     ; preds = %slice.4.clone.loop_header.dim.1.preheader.us, %slice.4.clone.loop_header.dim.2.preheader.us
  %slice.4.clone.invar_address.dim.1.08.us = phi i64 [ %lo_dim_1, %slice.4.clone.loop_header.dim.1.preheader.us ], [ %invar.inc3.us, %slice.4.clone.loop_header.dim.2.preheader.us ]
  %2 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, i64 %slice.4.clone.invar_address.dim.0.010.us, i64 %slice.4.clone.invar_address.dim.1.08.us, i64 40
  %3 = getelementptr inbounds [3 x [10000 x [24 x float]]], ptr %arg1, i64 0, i64 %slice.4.clone.invar_address.dim.0.010.us, i64 %slice.4.clone.invar_address.dim.1.08.us, i64 0
  %4 = load <8 x float>, ptr %2, align 32, !invariant.load !1, !noalias !5
  store <8 x float> %4, ptr %3, align 32, !alias.scope !5
  %5 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, i64 %slice.4.clone.invar_address.dim.0.010.us, i64 %slice.4.clone.invar_address.dim.1.08.us, i64 48
  %6 = getelementptr inbounds [3 x [10000 x [24 x float]]], ptr %arg1, i64 0, i64 %slice.4.clone.invar_address.dim.0.010.us, i64 %slice.4.clone.invar_address.dim.1.08.us, i64 8
  %7 = load <8 x float>, ptr %5, align 64, !invariant.load !1, !noalias !5
  store <8 x float> %7, ptr %6, align 32, !alias.scope !5
  %8 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, i64 %slice.4.clone.invar_address.dim.0.010.us, i64 %slice.4.clone.invar_address.dim.1.08.us, i64 56
  %9 = getelementptr inbounds [3 x [10000 x [24 x float]]], ptr %arg1, i64 0, i64 %slice.4.clone.invar_address.dim.0.010.us, i64 %slice.4.clone.invar_address.dim.1.08.us, i64 16
  %10 = load <8 x float>, ptr %8, align 32, !invariant.load !1, !noalias !5
  store <8 x float> %10, ptr %9, align 32, !alias.scope !5
  %invar.inc3.us = add nuw nsw i64 %slice.4.clone.invar_address.dim.1.08.us, 1
  %exitcond.not = icmp eq i64 %invar.inc3.us, %up_dim_1
  br i1 %exitcond.not, label %slice.4.clone.loop_header.dim.1.slice.4.clone.loop_exit.dim.1_crit_edge.us, label %slice.4.clone.loop_header.dim.2.preheader.us, !llvm.loop !8

slice.4.clone.loop_header.dim.1.slice.4.clone.loop_exit.dim.1_crit_edge.us: ; preds = %slice.4.clone.loop_header.dim.2.preheader.us
  %invar.inc.us = add nuw nsw i64 %slice.4.clone.invar_address.dim.0.010.us, 1
  %exitcond15.not = icmp eq i64 %invar.inc.us, %up_dim_0
  br i1 %exitcond15.not, label %return, label %slice.4.clone.loop_header.dim.1.preheader.us, !llvm.loop !10

return:                                           ; preds = %slice.4.clone.loop_header.dim.1.slice.4.clone.loop_exit.dim.1_crit_edge.us, %1
  ret ptr null
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 2}
!1 = !{}
!2 = !{i64 7680000}
!3 = !{i64 64}
!4 = !{i64 2880000}
!5 = !{!6}
!6 = !{!"result slice: {index:1, offset:0, size:2880000}", !7}
!7 = !{!"XLA host kernel slice.4.clone_kernel AA domain"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.unroll.disable"}
!10 = distinct !{!10, !9}
