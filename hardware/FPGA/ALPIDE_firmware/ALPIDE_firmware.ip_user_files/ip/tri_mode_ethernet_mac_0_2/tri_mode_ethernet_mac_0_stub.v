// Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2018.3 (lin64) Build 2405991 Thu Dec  6 23:36:41 MST 2018
// Date        : Tue Jan  7 18:00:24 2020
// Host        : lxele02.pd.infn.it running 64-bit CentOS Linux release 7.7.1908 (Core)
// Command     : write_verilog -force -mode synth_stub -rename_top tri_mode_ethernet_mac_0 -prefix
//               tri_mode_ethernet_mac_0_ tri_mode_ethernet_mac_0_stub.v
// Design      : tri_mode_ethernet_mac_0
// Purpose     : Stub declaration of top-level module interface
// Device      : xc7a35tcsg324-1
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
(* x_core_info = "tri_mode_ethernet_mac_0_block,Vivado 2018.3" *)
module tri_mode_ethernet_mac_0(glbl_rstn, rx_axi_rstn, tx_axi_rstn, rx_enable, 
  rx_statistics_vector, rx_statistics_valid, rx_mac_aclk, rx_reset, rx_axis_mac_tdata, 
  rx_axis_mac_tvalid, rx_axis_mac_tlast, rx_axis_mac_tuser, tx_enable, tx_ifg_delay, 
  tx_statistics_vector, tx_statistics_valid, tx_mac_aclk, tx_reset, tx_axis_mac_tdata, 
  tx_axis_mac_tvalid, tx_axis_mac_tlast, tx_axis_mac_tuser, tx_axis_mac_tready, pause_req, 
  pause_val, speedis100, speedis10100, mii_txd, mii_tx_en, mii_tx_er, mii_rxd, mii_rx_dv, 
  mii_rx_er, mii_rx_clk, mii_tx_clk, rx_configuration_vector, tx_configuration_vector)
/* synthesis syn_black_box black_box_pad_pin="glbl_rstn,rx_axi_rstn,tx_axi_rstn,rx_enable,rx_statistics_vector[27:0],rx_statistics_valid,rx_mac_aclk,rx_reset,rx_axis_mac_tdata[7:0],rx_axis_mac_tvalid,rx_axis_mac_tlast,rx_axis_mac_tuser,tx_enable,tx_ifg_delay[7:0],tx_statistics_vector[31:0],tx_statistics_valid,tx_mac_aclk,tx_reset,tx_axis_mac_tdata[7:0],tx_axis_mac_tvalid,tx_axis_mac_tlast,tx_axis_mac_tuser[0:0],tx_axis_mac_tready,pause_req,pause_val[15:0],speedis100,speedis10100,mii_txd[3:0],mii_tx_en,mii_tx_er,mii_rxd[3:0],mii_rx_dv,mii_rx_er,mii_rx_clk,mii_tx_clk,rx_configuration_vector[79:0],tx_configuration_vector[79:0]" */;
  input glbl_rstn;
  input rx_axi_rstn;
  input tx_axi_rstn;
  output rx_enable;
  output [27:0]rx_statistics_vector;
  output rx_statistics_valid;
  output rx_mac_aclk;
  output rx_reset;
  output [7:0]rx_axis_mac_tdata;
  output rx_axis_mac_tvalid;
  output rx_axis_mac_tlast;
  output rx_axis_mac_tuser;
  output tx_enable;
  input [7:0]tx_ifg_delay;
  output [31:0]tx_statistics_vector;
  output tx_statistics_valid;
  output tx_mac_aclk;
  output tx_reset;
  input [7:0]tx_axis_mac_tdata;
  input tx_axis_mac_tvalid;
  input tx_axis_mac_tlast;
  input [0:0]tx_axis_mac_tuser;
  output tx_axis_mac_tready;
  input pause_req;
  input [15:0]pause_val;
  output speedis100;
  output speedis10100;
  output [3:0]mii_txd;
  output mii_tx_en;
  output mii_tx_er;
  input [3:0]mii_rxd;
  input mii_rx_dv;
  input mii_rx_er;
  input mii_rx_clk;
  input mii_tx_clk;
  input [79:0]rx_configuration_vector;
  input [79:0]tx_configuration_vector;
endmodule
