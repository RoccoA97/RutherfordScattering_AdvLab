library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use IEEE.STD_LOGIC_UNSIGNED.ALL;
library unisim;
use unisim.vcomponents.all;

use work.ipbus.all;
use work.ipbus_reg_types.all;
use work.ipbus_decode_ALPIDE_regs.all;

entity top_level is 
  port(
    clk_base_xc7a_i : in std_logic; --Arty base clock 100MHz

    xc7a_rst_i  : in std_logic;         -- hw reset input
    xc7a_led0_o : out std_logic;        -- heart beat - system clock

    -- IPBUS
    mii_tx_en_o  : out std_logic;
    mii_tx_clk_i : in  std_logic;
    mii_txd_o    : out std_logic_vector(3 downto 0);
    mii_rx_clk_i : in  std_logic;
    mii_rx_dv_i  : in  std_logic;
    mii_rxd_i    : in  std_logic_vector(3 downto 0);
    clk_phy_o    : out std_logic;
    rst_n_phy_o  : out std_logic;
    
     -- ALPIDE Signals
	CTRL_io : inout STD_LOGIC;  --data pin
    DCLK_o : out STD_LOGIC;    --MASTER clk 40MHz
    rst_n_io_o : out STD_LOGIC := '1' ;    --ALPIDE resetn 
    pwr_enable_o : out STD_LOGIC ; --ALPIDE power enable
    rstn_i : in STD_LOGIC := '1';  --Arty resetn (red one)    
    init_led_o : out STD_LOGIC ;    --initialized status
    power_led_o : out STD_LOGIC ;   --powered status
    busy_led_o : out STD_LOGIC ;    --busy status
    read_err_led_o : out STD_LOGIC ;   --stop bit not detected
    idle_err_led_o : out STD_LOGIC ;   --high input not detected on idle phase
    slave_err_led_o : out STD_LOGIC ;    --if slave drive the line for 51 or more clk cycles 
    id_err_led_o : out STD_LOGIC    --different chip_id recieved  
    
     
    );
end top_level;

architecture rtl of top_level is


-- clocks related signals
  signal s_sysclk          : std_logic;
  signal s_sysclk_x4       : std_logic;
  signal s_sysclk_x2       : std_logic;
  signal s_locked_tx       : std_logic;
  signal s_clk_200         : std_logic;

  signal reset_i : std_logic;
  
  --resetn through ipBUS
  signal rstn_s             : STD_LOGIC := '1';
  signal rstn_ipBUS         : STD_LOGIC ;


-- heart beat
  signal u_led_count  : unsigned(24 downto 0);
  signal s_led_count  : std_logic_vector(24 downto 0);

----IPBUS signals
  signal ipb_clk, locked, rst_125, rstn_125, rst_ipb, onehz : std_logic;
  signal mac_tx_data, mac_rx_data : std_logic_vector(7 downto 0);
  signal mac_tx_valid, mac_tx_last, mac_tx_error,
    mac_tx_ready,mac_rx_valid, mac_rx_last, mac_rx_error : std_logic;
  signal ipb_master_out                                  : ipb_wbus;
  signal ipb_master_in                                   : ipb_rbus;
  signal sys_rst : std_logic;
  signal s_ip_addr                                       : std_logic_vector(31 downto 0);
  signal s_mac_addr                                      : std_logic_vector(47 downto 0);

  signal ipbw: ipb_wbus_array(N_SLAVES - 1 downto 0);
  signal ipbr: ipb_rbus_array(N_SLAVES - 1 downto 0);
  
  signal ctrl : ipb_reg_v(0 downto 0); --control and status register
  signal stat : ipb_reg_v(0 downto 0);
  signal q_s_cmd : ipb_reg_v(4 downto 0);
  
  signal d_s_data : std_logic_vector(31 downto 0);
  signal q_s_data : std_logic_vector(31 downto 0);
  signal we_s_data  : std_logic;	
  signal addr_s_data : std_logic_vector(9 downto 0);
  
  --ALPIDE_carrier signals
  signal ALPIDE_clk_100MHz : STD_LOGIC ;
  signal ALPIDE_op_sw : STD_LOGIC_VECTOR ( 3 downto 0 ) ;
  signal ALPIDE_strt : STD_LOGIC ;
  signal ALPIDE_rstn : STD_LOGIC ;
  
  signal ALPIDE_CTRL : STD_LOGIC ;
  signal ALPIDE_DCLK : STD_LOGIC ;
  signal ALPIDE_rst_n_io : STD_LOGIC ;
  signal ALPIDE_pwr_enable : STD_LOGIC ;
  
  signal ALPIDE_RR_addr : STD_LOGIC_VECTOR ( 15 downto 0 ) ;	
  signal ALPIDE_OP_command : STD_LOGIC_VECTOR ( 7 downto 0 ) ;	
  signal ALPIDE_CHIP_id : STD_LOGIC_VECTOR ( 7 downto 0 ) :=x"12" ;		
  signal ALPIDE_WR_addr : STD_LOGIC_VECTOR ( 15 downto 0 ) ;	
  signal ALPIDE_WR_data : STD_LOGIC_VECTOR ( 15 downto 0 ) ;	
  
  signal ALPIDE_data_reg : STD_LOGIC_VECTOR ( 15 downto 0 ) :=x"FAFA" ;	--addr 0x200
  signal ALPIDE_data_FIFO : STD_LOGIC_VECTOR ( 31 downto 0 ) := x"F0F0F0F0" ;	--base addr 0x000
  signal ALPIDE_d_reg_we : STD_LOGIC := '0' ;   --data on register valid
  signal ALPIDE_d_FIFO_we : STD_LOGIC := '0' ;  --read out data valid
						
  signal ALPIDE_init_led : STD_LOGIC ;
  signal ALPIDE_power_led : STD_LOGIC ;
  signal ALPIDE_busy_led : STD_LOGIC ;
  
  signal ALPIDE_read_err : STD_LOGIC ;
  signal ALPIDE_idle_err : STD_LOGIC ;
  signal ALPIDE_slave_err : STD_LOGIC ;
  signal ALPIDE_id_err : STD_LOGIC ;
  signal ALPIDE_ro_stop : STD_LOGIC ;   --stop read out
  signal ALPIDE_mem_read, mem_read1, mem_read2 : STD_LOGIC ;    --ipbus memory read (given by software)
  signal ALPIDE_mem_readable : STD_LOGIC ;  --status given by Arty
  signal ALPIDE_lkd : STD_LOGIC ;   --PLL locked
  
  signal strt : STD_LOGIC ;
  
  --signal to detect rising edge
  signal d_reg_we1 : STD_LOGIC ;
  signal d_reg_we2 : STD_LOGIC ;
  signal d_FIFO_we1 : STD_LOGIC ;
  signal FIFO_we1 : STD_LOGIC ;
  signal FIFO_we2 : STD_LOGIC ;
  signal d_FIFO_we2 : STD_LOGIC ;
  
  --critical error on read out or read register
  signal error_s : STD_LOGIC ;
  
  -- command console and programm signal
  type state is (s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10);
  signal cc_state, FIFO_state : state := s0 ;
  
  -- PWM signal for RGB LED
  signal pwm_signal : STD_LOGIC;
  
  -- FIFO signal
  signal FIFO_rst : STD_LOGIC;
  signal FIFO_wr_clk : STD_LOGIC;
  signal FIFO_rd_clk : STD_LOGIC;
  signal FIFO_din : STD_LOGIC_VECTOR(23 DOWNTO 0);
  signal FIFO_wr_en : STD_LOGIC;
  signal FIFO_rd_en : STD_LOGIC;
  signal FIFO_dout : STD_LOGIC_VECTOR(23 DOWNTO 0);
  signal FIFO_full : STD_LOGIC;
  signal FIFO_prog_full : STD_LOGIC;
  signal FIFO_empty : STD_LOGIC;
  
  constant IO_DATA_WIDTH_status : integer := 13;
  constant IO_DATA_WIDTH_control : integer := 7;
  constant IO_DATA_WIDTH_CMD_regs : integer := 64;
	
	--Dual port ram    ethenet <--> ipbus <--> vhdl code
    component ipbus_dpram
            generic(
                ADDR_WIDTH: natural
            );
            port(
                clk: in std_logic;
                rst: in std_logic;
                ipb_in: in ipb_wbus;
                ipb_out: out ipb_rbus;
                rclk: in std_logic;
                we: in std_logic := '0';
                d: in std_logic_vector(31 downto 0) := (others => '0');
                q: out std_logic_vector(31 downto 0);
                addr: in std_logic_vector(9 downto 0)
            );	
    end component;
    
    -- control and status registers
    component ipbus_ctrlreg_v
        generic(
            N_CTRL: natural := 1;
            N_STAT: natural := 1;
            SWAP_ORDER: boolean := false
            );
        port(
            clk: in std_logic;
            reset: in std_logic;
            ipbus_in: in ipb_wbus;
            ipbus_out: out ipb_rbus;
            d: in ipb_reg_v(N_STAT - 1 downto 0) := (others => (others => '0'));
            q: out ipb_reg_v(N_CTRL - 1 downto 0);
            qmask: in ipb_reg_v(N_CTRL - 1 downto 0) := (others => (others => '1'));
            stb: out std_logic_vector(N_CTRL - 1 downto 0)
            );
    end component;
    
    -- registers
    component ipbus_reg_v
        generic(
            N_REG: positive := 1
            );
        port(
            clk: in std_logic;
            reset: in std_logic;
            ipbus_in: in ipb_wbus;
            ipbus_out: out ipb_rbus;
            q: out ipb_reg_v(N_REG - 1 downto 0);
            qmask: in ipb_reg_v(N_REG - 1 downto 0) := (others => (others => '1'));
            stb: out std_logic_vector(N_REG - 1 downto 0)
           );
    end component;
    
    -- Fifo  
    component READOUT_FIFO
      PORT (
        rst : IN STD_LOGIC;
        wr_clk : IN STD_LOGIC;
        --rd_clk : IN STD_LOGIC;
        din : IN STD_LOGIC_VECTOR(23 DOWNTO 0);
        wr_en : IN STD_LOGIC;
        rd_en : IN STD_LOGIC;
        dout : OUT STD_LOGIC_VECTOR(23 DOWNTO 0);
        full : OUT STD_LOGIC;
        prog_full : OUT STD_LOGIC;
        empty : OUT STD_LOGIC
      );
    end component;
    
    -- CDC synchronizer
    component CDC_module
        Generic (IO_DATA_WIDTH : integer := 1;
                sync_FF : integer := 2);
        Port ( I_s : in STD_LOGIC_VECTOR (IO_DATA_WIDTH-1 downto 0);
               I_Clk : in STD_LOGIC;
               O_Clk : in STD_LOGIC;
               O_s : out STD_LOGIC_VECTOR (IO_DATA_WIDTH-1 downto 0)
           );
    end component;
    
    -- ALPIDE module
    component ALPIDE_Carrier
       Generic (MAX_SLV_CLK : STD_LOGIC_VECTOR(7 downto 0) := x"33");
       
       Port ( clk_100MHz : in STD_LOGIC;   --sys clock 100MHz
    
           CTRL : inout STD_LOGIC;  --data pin
           DCLK : out STD_LOGIC;    --MASTER clk
           pwr_enable : out STD_LOGIC ; --ALPIDE power enable
           rst_n_io : out STD_LOGIC := '1' ;    --ALPIDE resetn 
           
           op_sw : in STD_LOGIC_VECTOR (3 downto 0) := "0000";    --16 selectable options
           A_strt : in STD_LOGIC;  --validate option
           rstn : in STD_LOGIC := '1';  --Arty resetn
           
           RR_addr : in STD_LOGIC_VECTOR (15 downto 0) ; 
           OP_command : in STD_LOGIC_VECTOR (7 downto 0) ; 
           CHIP_id : in STD_LOGIC_VECTOR (7 downto 0) ;   --Chip identifier (MULTICAST ID is 0F) set via ALPIDE switches
           WR_addr : in STD_LOGIC_VECTOR (15 downto 0) ;
           WR_data : in STD_LOGIC_VECTOR (15 downto 0) ;
           
           data_reg : out STD_LOGIC_VECTOR (15 downto 0) := x"FFFF" ; --data read on selected register
           data_FIFO : out STD_LOGIC_VECTOR (31 downto 0) := x"00000000" ;   --data read on FIFO
		   d_reg_we : out STD_LOGIC ;   --data register out write enable
           d_FIFO_we : out STD_LOGIC ;   --data FIFO out write enable
           ro_stop : in STD_LOGIC ;        --stop continous read out
           ro_FIFO_full : in STD_LOGIC ;    --Data FIFO full signal  
            
           init_led : out STD_LOGIC ;
           power_led : out STD_LOGIC ;
           busy_led : out STD_LOGIC ;
           lkd : out STD_LOGIC ;
           
           read_err : out STD_LOGIC ;   --stop bit not detected
           idle_err : out STD_LOGIC ;   --high input not detected on idle phase
           slave_err : out STD_LOGIC ;    --if slave drive the line for 51 or more clk cycles 
           chip_id_err: out STD_LOGIC );    --different chip_id recieved     
    end component;

------------------------------------------------------------------------------------
------------------------------------------DEBUG-------------------------------------

   

begin
-------------------------------------IP, MAC and GCU ID---------------------------------------
  s_ip_addr  <= X"0A0A0A64"; -- 10.10.10.100
  s_mac_addr <= X"020ddba11599";

  reset_i <= rst_ipb;
  rst_n_phy_o <= not rst_ipb;
  
------------------------------------------------------------------------------------
-------------------Physical resetn AND with software resetn(ipBUS)---------------------
    
    rstn_s <= rstn_i AND rstn_ipBUS;

------------------------------------------------------------------------------------
-------------------------------PLL and system clock---------------------------------
  Inst_system_clocks : entity work.system_clocks
    port map(
      sysclk_p    => clk_base_xc7a_i,
      clko_ipb    => ipb_clk,           -- 31.25 MHz
      sysclk_o    => s_sysclk,          -- 62.5 MHz
      sysclk_x2_o => s_sysclk_x2,       -- 125 MHz
      sysclk_x4_o => s_sysclk_x4,       -- 250 MHz
      phy_clk_o   => clk_phy_o,         -- 10 MHz
      clk_200_o   => s_clk_200,
      locked      => s_locked_tx,
      nuke        => sys_rst,
      rsto_125    => rst_125,
      rsto_ipb    => rst_ipb,
      onehz       => onehz
      );

     rstn_125 <= not rst_125;
------------------------------------------------------------------------------------
----------------------------------------IPBUS---------------------------------------

  eth_mac_block_1 : entity work.tri_mode_ethernet_mac_0_fifo_block
    port map (
    --  gtx_clk                 => s_sysclk_x2,  -- 125 MHz
      glbl_rstn               => rstn_125,
      rx_axi_rstn             => rstn_125,
      tx_axi_rstn             => rstn_125,
      rx_mac_aclk             => open,
      rx_reset                => open,
      rx_statistics_vector    => open,
      rx_statistics_valid     => open,
      rx_fifo_clock           => s_sysclk_x2,  -- 125 MHz
      rx_fifo_resetn          => rstn_125,
      rx_axis_fifo_tdata      => mac_rx_data,
      rx_axis_fifo_tvalid     => mac_rx_valid,
      rx_axis_fifo_tready     => '1',
      rx_axis_fifo_tlast      => mac_rx_last,
      tx_mac_aclk             => open,
      tx_reset                => open,
      tx_ifg_delay            => X"00",
      tx_statistics_vector    => open,
      tx_statistics_valid     => open,
      tx_fifo_clock           => s_sysclk_x2,  -- 125 MHz
      tx_fifo_resetn          => rstn_125,
      tx_axis_fifo_tdata      => mac_tx_data,
      tx_axis_fifo_tvalid     => mac_tx_valid,
      tx_axis_fifo_tready     => mac_tx_ready,
      tx_axis_fifo_tlast      => mac_tx_last,
      pause_req               => '0',
      pause_val               => X"0000",
      mii_txd                 => mii_txd_o,
      mii_tx_en               => mii_tx_en_o,
      mii_tx_er               => open,
      mii_rxd                 => mii_rxd_i,
      mii_rx_dv               => mii_rx_dv_i,
      mii_rx_er               => '0',
      mii_rx_clk              => mii_rx_clk_i,
      mii_tx_clk              => mii_tx_clk_i,
      rx_configuration_vector => X"0000_0000_0000_0000_1012",
      tx_configuration_vector => X"0000_0000_0000_0000_1012");

-- ipbus control logic

  ipbus : entity work.ipbus_ctrl
    port map(
      mac_clk      => s_sysclk_x2,      -- 125 MHz
      rst_macclk   => rst_125,
      ipb_clk      => ipb_clk,          -- 31.25 MHz
      rst_ipb      => rst_ipb,
      mac_rx_data  => mac_rx_data,
      mac_rx_valid => mac_rx_valid,
      mac_rx_last  => mac_rx_last,
      mac_rx_error => mac_rx_error,
      mac_tx_data  => mac_tx_data,
      mac_tx_valid => mac_tx_valid,
      mac_tx_last  => mac_tx_last,
      mac_tx_error => mac_tx_error,
      mac_tx_ready => mac_tx_ready,
      ipb_out      => ipb_master_out,
      ipb_in       => ipb_master_in,
      mac_addr     => s_mac_addr,
      ip_addr      => s_ip_addr
      );

  fabric: entity work.ipbus_fabric_sel
    generic map(
      NSLV => N_SLAVES,
      SEL_WIDTH => IPBUS_SEL_WIDTH)
    port map(
      ipb_in => ipb_master_out,
      ipb_out => ipb_master_in,
      sel => ipbus_sel_ALPIDE_regs(ipb_master_out.ipb_addr),
      ipb_to_slaves => ipbw,
      ipb_from_slaves => ipbr
    );
    
    --status and control register (BUSY, POWERED, INIT, FIFO_full, FIFO_empty, and errors)
    CTRL_STATUS : ipbus_ctrlreg_v
        generic map(
            N_CTRL => 1,
            N_STAT => 1)
        port map(
            clk         => ipb_clk,
            reset       => rst_ipb,
            ipbus_in    => ipbw(N_SLV_CSR),
            ipbus_out   => ipbr(N_SLV_CSR),
            d           => stat,    --(BUSY, POWERED, INIT, FIFO_full, FIFO_empty)
            q           => ctrl     --strt, op_sw
        );
    
    ALPIDE_status : CDC_module
        generic map(
            IO_DATA_WIDTH => IO_DATA_WIDTH_status)
        port map(
            I_s(0)      => ALPIDE_power_led,
            I_s(1)      => ALPIDE_init_led ,
            I_s(2)      => ALPIDE_busy_led,
            I_s(3)      => FIFO_full,
            I_s(4)      => FIFO_empty,
            I_s(5)      => FIFO_prog_full,
            I_s(6)      => ALPIDE_mem_readable,
            I_s(7)      => ALPIDE_slave_err,
            I_s(8)      => ALPIDE_idle_err,
            I_s(9)      => ALPIDE_read_err,
            I_s(10)     => ALPIDE_id_err,
            I_s(11)     => ALPIDE_lkd,
            I_s(12)     => s_locked_tx,
            I_Clk       => ALPIDE_DCLK,
            O_Clk       => ipb_clk,
            O_s         => stat(0)(IO_DATA_WIDTH_status-1 downto 0)
            );   
	--stat(0)(0) <= ALPIDE_power_led ;
	--stat(0)(1) <= ALPIDE_init_led ;
	--stat(0)(2) <= ALPIDE_busy_led ;
	--stat(0)(3) <= FIFO_full ;
	--stat(0)(4) <= FIFO_empty ;
	--stat(0)(5) <= FIFO_prog_full ;
	--stat(0)(6) <= ALPIDE_mem_readable ;
	--stat(0)(7) <= ALPIDE_slave_err ;
	--stat(0)(8) <= ALPIDE_idle_err ;
	--stat(0)(9) <= ALPIDE_read_err ;
	--stat(0)(10) <= ALPIDE_id_err ;
	--stat(0)(11) <= ALPIDE_lkd ;
	--stat(0)(12) <= s_locked_tx ;
	
	ALPIDE_control : CDC_module
        generic map(
            IO_DATA_WIDTH => IO_DATA_WIDTH_control)
        port map(
            I_s             => ctrl(0)(IO_DATA_WIDTH_control-1 downto 0),
            I_Clk           => ipb_clk,
            O_Clk           => ALPIDE_DCLK,
            O_s(0)          => ALPIDE_strt,
            O_s(4 downto 1) => ALPIDE_op_sw,
            O_s(5)          => ALPIDE_ro_stop,
            O_s(6)          => ALPIDE_mem_read
            --O_s/7)        => rstn_ipBUS
            );   
	--ALPIDE_strt <= ctrl(0)(0) ;
	--ALPIDE_op_sw <= ctrl(0)(4 downto 1) ;
	--ALPIDE_ro_stop <= ctrl(0)(5) ;
	--ALPIDE_mem_read <= ctrl(0)(6) ;
	rstn_ipBUS   <= ctrl(0)(7) ;
    

	
-- Command reg
    CMD_reg : ipbus_reg_v
        generic map(N_REG => 5)
        port map(
            clk         => ipb_clk,
            reset       => rst_ipb,
            ipbus_in    => ipbw(N_SLV_CMD_ADDR),
            ipbus_out   => ipbr(N_SLV_CMD_ADDR),
            q           => q_s_cmd,
            stb         => open
	   );
	   
	ALPIDE_CMD_regs : CDC_module
        generic map(
            IO_DATA_WIDTH => IO_DATA_WIDTH_CMD_regs)
        port map(
            I_s (15 downto 0)       => q_s_cmd(0)(15 downto 0),
            I_s (23 downto 16)      => q_s_cmd(1)(7 downto 0),
            I_s (31 downto 24)      => q_s_cmd(2)(7 downto 0),
            I_s (47 downto 32)      => q_s_cmd(3)(15 downto 0),
            I_s (63 downto 48)      => q_s_cmd(4)(15 downto 0),
            I_Clk                   => ipb_clk,
            O_Clk                   => ALPIDE_DCLK,
            O_s(15 downto 0)        => ALPIDE_RR_addr,
            O_s(23 downto 16)       => ALPIDE_OP_command,
            O_s(31 downto 24)       => ALPIDE_Chip_ID,
            O_s(47 downto 32)       => ALPIDE_WR_addr,
            O_s(63 downto 48)       => ALPIDE_WR_data
            );     
    
    --ALPIDE_RR_addr <= q_s_cmd(0)(15 downto 0) ;
	--ALPIDE_OP_command <= q_s_cmd(1)(7 downto 0) ;
	--ALPIDE_Chip_ID <= q_s_cmd(2)(7 downto 0) ;
	--ALPIDE_WR_addr <= q_s_cmd(3)(15 downto 0) ;
	--ALPIDE_WR_data <= q_s_cmd(4)(15 downto 0) ;
	
	-- Data
  Data_reg: ipbus_dpram 
	generic map( ADDR_WIDTH => 10)
	port map(
            clk        => ipb_clk,
            rst        => rst_ipb,
            ipb_in     => ipbw(N_SLV_DATA),
            ipb_out    => ipbr(N_SLV_DATA),
		    rclk       => ALPIDE_DCLK,
		    we         => we_s_data,
		    d          => d_s_data,
		    q          => open,
		    addr       => addr_s_data
	);
	
	--FIFO_DATA
	FIFO : READOUT_FIFO
	port map
	   (rst        =>  FIFO_rst,
	   wr_clk      =>  ALPIDE_DCLK,
	   --rd_clk      =>  ALPIDE_DCLK_buff,
	   din         =>  FIFO_din,
	   wr_en       =>  FIFO_wr_en,
	   rd_en       =>  FIFO_rd_en,
	   dout        =>  FIFO_dout,
	   full        =>  FIFO_full,
	   prog_full   =>  FIFO_prog_full,
	   empty       =>  FIFO_empty );
        
    FIFO_rst <= (not rstn_s) or ALPIDE_ro_stop; --this one should be changed
    
-----------------------------------------------------------------
--- ALPIDE Carrirer instatiation
-----------------------------------------------------------------

   ALPIDE_reader : ALPIDE_Carrier 
                    generic map (MAX_SLV_CLK    => x"33")
                    
					port map (	clk_100MHz 		=> clk_base_xc7a_i,
								CTRL       		=> CTRL_io,
								DCLK       		=> ALPIDE_DCLK,
								pwr_enable 		=> pwr_enable_o,
								rst_n_io   		=> rst_n_io_o,
								op_sw      		=> ALPIDE_op_sw,
								A_strt        	=> ALPIDE_strt,
								rstn       		=> ALPIDE_rstn,
								RR_addr			=> ALPIDE_RR_addr,
								OP_command		=> ALPIDE_OP_command,
								CHIP_id			=> ALPIDE_CHIP_id,
								WR_addr			=> ALPIDE_WR_addr,
								WR_data			=> ALPIDE_WR_data,
								data_reg 		=> ALPIDE_data_reg,
								data_FIFO		=> ALPIDE_data_FIFO,
								d_reg_we        => ALPIDE_d_reg_we,
								d_FIFO_we       => ALPIDE_d_FIFO_we,
								ro_stop         => ALPIDE_ro_stop,
								ro_FIFO_full    => FIFO_full,
								init_led   		=> ALPIDE_init_led,
								power_led  		=> ALPIDE_power_led,
								busy_led   		=> ALPIDE_busy_led,
								lkd             => ALPIDE_lkd,
								read_err        => ALPIDE_read_err,
								idle_err        => ALPIDE_idle_err,
								slave_err       => ALPIDE_slave_err,
								chip_id_err     => ALPIDE_id_err 
								);

	ALPIDE_clk_100MHz <= clk_base_xc7a_i ;	--base clk
	pwr_enable_o <= ALPIDE_pwr_enable ;
	rst_n_io_o <= ALPIDE_rst_n_io ;
	ALPIDE_rstn <= rstn_s ;
	DCLK_o <= ALPIDE_DCLK ;
	init_led_o <= ALPIDE_init_led ;
	power_led_o <= ALPIDE_power_led ;
	busy_led_o <= ALPIDE_busy_led ;
	read_err_led_o <= ALPIDE_read_err and pwm_signal ;     --1 rgb led red
	idle_err_led_o <= ALPIDE_idle_err and pwm_signal ;     --2 rgb led red
	slave_err_led_o <= ALPIDE_slave_err and pwm_signal ;   --3 rgb led red
	id_err_led_o <= ALPIDE_id_err and pwm_signal ;         --0 rgb led red

------------------------------------------------------------------------------------
---------------------------------------heart beat-----------------------------------
  p_heart_beat_counter : process(s_sysclk)
  begin
    if (rising_edge(s_sysclk)) then
        u_led_count <= u_led_count + 1;
    end if;
  end process p_heart_beat_counter;
  s_led_count <= std_logic_vector(u_led_count);

-----------------------LEDs output ---------------------
  xc7a_led0_o <= s_led_count(24);       -- heart beat 
  
  
  process (ALPIDE_DCLK, ALPIDE_d_reg_we, ALPIDE_d_FIFO_we, ALPIDE_mem_read, rstn_s) is  --detect signals' rising edge 
  begin
  
    if rstn_s = '0' then
        d_reg_we2 <= '0' ;
        d_reg_we1 <= '0' ;
        d_FIFO_we2 <= '0' ;
        d_FIFO_we1 <= '0' ;
        mem_read2 <= '0' ;
        mem_read1 <= '0' ;
    elsif rising_edge(ALPIDE_DCLK) then
        d_reg_we2 <= d_reg_we1 ;
        d_reg_we1 <= ALPIDE_d_reg_we ;
        d_FIFO_we2 <= d_FIFO_we1 ;
        d_FIFO_we1 <= ALPIDE_d_FIFO_we ;
        mem_read2 <= mem_read1 ;
        mem_read1 <= ALPIDE_mem_read ;
    end if;
  
  end process;
  
  error_s <= ALPIDE_read_err or ALPIDE_slave_err ;
   
  
  load_store_data : process (ALPIDE_DCLK, rstn_s)
  
  variable cnt : STD_LOGIC_VECTOR ( 11 downto 0 ) := x"000" ;
  variable w_clk : STD_LOGIC_VECTOR ( 15 downto 0 ) := x"0000" ;
  
  begin
	if rstn_s = '0' then   -- Asynchrouns reset
		cnt := x"000" ;
		w_clk := x"0000" ;
		cc_state <= s0 ;
	elsif rising_edge(ALPIDE_DCLK) then	--which clk to use?
	   case cc_state is
	   when s0 =>
	       we_s_data <= '0' ;
	       FIFO_rd_en <= '0' ;
	       if ALPIDE_busy_led = '1' and error_s /= '1' then
	           if ALPIDE_op_sw = x"4" then --load data of selected register
	               cc_state <= s1 ;
	               addr_s_data <= "1000000000" ;   --dec 1024
	           elsif  ALPIDE_op_sw= x"5" then  --continous read out
	               addr_s_data <= "0000000000" ;   --dpram base address
	               cc_state <= s3 ;
	           end if;
            end if; 
	   when s1 =>  --load the 16 bit data on dpram
	       we_s_data <= '0' ;
	       if error_s = '1' then
	           cc_state <= s0 ;
	       else
               if d_reg_we1 = '1' and d_reg_we2 = '0'  then
                   addr_s_data <= "1000000000" ;    
                   we_s_data <= '1' ;
                   cc_state <= s2 ;
               end if;
	       end if;    
	   when s2 =>
	       we_s_data <= '1' ;
	       d_s_data <= x"0000" & ALPIDE_data_reg ;
	       if ALPIDE_busy_led = '0' then   --wait until ALPIDE busy = '0'
	           cc_state <= s0 ;
	       end if;
	   when s3 =>
	       if FIFO_prog_full = '1' then    --wait until FIFO has 512 data stored
	           FIFO_rd_en <= '1' ;
	           cc_state <= s4 ;
	       end if;
	       if ALPIDE_ro_stop = '1' then
	           cc_state <= s7 ;
	           ALPIDE_mem_readable <= '0' ;
	           we_s_data <= '0' ;
	           FIFO_rd_en <= '0' ;
	       end if;
	   when s4 =>
	       FIFO_rd_en <= '1' ;
	       cc_state <= s5 ;    
	   when s5 =>
	       we_s_data <= '1' ;
	       FIFO_rd_en <= '1' ;
	       --cnt := x"000" ;
	       addr_s_data <= "0000000000"  ; --baseaddress x"000"
	       d_s_data <= x"00" & FIFO_dout ;
	       cc_state <= s6 ;
	   when s6 =>
	       if addr_s_data < x"1FD" then
	           we_s_data <= '1' ;
	           FIFO_rd_en <= '1' ;
	           d_s_data <= x"00" & FIFO_dout ;
	           --cnt := cnt + x"001" ;
	           addr_s_data <= addr_s_data + "0000000001"  ; 
	           ALPIDE_mem_readable <= '0' ;
	       elsif addr_s_data = x"1FD" or addr_s_data = x"1FE"  then
	           we_s_data <= '1' ;
	           FIFO_rd_en <= '0' ;
	           d_s_data <= x"00" & FIFO_dout ;
	           --cnt := cnt + x"001" ;
	           addr_s_data <= addr_s_data + "0000000001"  ; 
	           ALPIDE_mem_readable <= '0' ;
	       else
	           --cnt := x"000" ;
	           we_s_data <= '0' ;
	           FIFO_rd_en <= '0' ;
	           ALPIDE_mem_readable <= '1' ;
	           cc_state <= s7 ;
	       end if;
	       if error_s = '1' then
	           cc_state <= s0 ;
	           cnt := x"000" ;
	       end if;
	   when s7 =>
	       addr_s_data <= "0000000000" ;
	       we_s_data <= '0' ;
	       FIFO_rd_en <= '0' ;
	       if mem_read1 = '1' and mem_read2 = '0' then
	           ALPIDE_mem_readable <= '0' ;
	           cc_state <= s3 ;
	       end if;
	       if ALPIDE_ro_stop = '1' then
	           cc_state <= s8 ;
	           ALPIDE_mem_readable <= '0' ;
	           we_s_data <= '0' ;
	           FIFO_rd_en <= '0' ;
	       end if;    
	   when s8 =>
	       if ALPIDE_busy_led = '0' then
	           cc_state <= s0 ;
	       end if;           
	   when others =>
	       cc_state <= s0 ; 
	   end case;
	end if;
  
  
  end process load_store_data ;
  
  
  
      ----------------------------------------------------
      --Load data on FIFO
      ----------------------------------------------------
    load_FIFO_data : process (ALPIDE_DCLK, rstn_s)
      
    begin
    if rstn_s = '0' then
        FIFO_wr_en <= '0' ;
        FIFO_state <= s0 ;
    elsif rising_edge(ALPIDE_DCLK) then
        case FIFO_state is
        when s0 =>
            if d_FIFO_we1 = '1' and d_FIFO_we2 = '0' and FIFO_full /= '1' then
                FIFO_wr_en <= '0' ;
                FIFO_din <= ALPIDE_data_FIFO (23 downto 0) ;
                FIFO_state <= s1 ; 
            else
                FIFO_wr_en <= '0' ;         
            end if;
        when s1 =>
            FIFO_wr_en <= '1' ;
            FIFO_din <= ALPIDE_data_FIFO (23 downto 0) ;
            FIFO_state <= s0 ;
        when others =>
            FIFO_state <= s0 ;
        end case;    
    end if;
    end process load_FIFO_DATA; 
  
    ---------------------------------------------------
    --PWM RGB LED (errors' color code)
    ---------------------------------------------------
    PWM_led : process(s_sysclk,rstn_s) is
    
    variable pwm_cnt : STD_LOGIC_VECTOR( 9 downto 0 ) := "0000000000" ;
    
    begin
        
        if rstn_s = '0' then
            pwm_cnt := "0000000000" ;
            pwm_signal <= '0' ; 
        elsif rising_edge(s_sysclk) then
            if pwm_cnt < "0000010000" then
                pwm_cnt := pwm_cnt + "0000000001" ;
                pwm_signal <= '1' ;
            else
                pwm_cnt := pwm_cnt + "0000000001" ;
                pwm_signal <= '0' ;
            end if;  
        end if;
        
    end process PWM_led;
   
  
end rtl;


--read_out not optimal ( data loss might occour)    -----FIXED
--put PWM signal on top module  -----DONE 
--change xml file to accomodate ctrl reg 
--change name of status variables
--change dpram for read reg(use one for all instead of 2)
--read out issues, data loss if read out is relaunch