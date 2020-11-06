----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 08.05.2020 00:05:33
-- Design Name: 
-- Module Name: RW_PLL - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

Library UNISIM;
use UNISIM.vcomponents.all;

entity RW_PLL is
    Port (  clk_in1 : in STD_LOGIC; --input clk 100MHz
            rstn : in STD_LOGIC;    --resetn
            clk_out1 : out STD_LOGIC;   --out clk 40MHz
            clk_out2 : out STD_LOGIC;   --out clk 40MHz 90° out of phase 
            locked : out STD_LOGIC);    --locked signal
end RW_PLL;

architecture Behavioral of RW_PLL is

--buffer signals
signal clk_fout, clk_fout_buff, reset, lkd : STD_LOGIC ;
signal clk_out1_mmcm, clk1_en : STD_LOGIC ;
signal clk_out2_mmcm, clk2_en : STD_LOGIC ;

    
begin
    -- MMCME2_BASE: Base Mixed Mode Clock Manager
    -- 7 Series
    -- Xilinx HDL Libraries Guide, version 2012.2
    MMCME2_BASE_inst : MMCME2_BASE
    generic map (
    BANDWIDTH => "OPTIMIZED", -- Jitter programming (OPTIMIZED, HIGH, LOW)
    CLKFBOUT_MULT_F => 10.0, -- Multiply value for all CLKOUT (2.000-64.000).
    CLKFBOUT_PHASE => 0.0, -- Phase offset in degrees of CLKFB (-360.000-360.000).
    CLKIN1_PERIOD => 10.0, -- Input clock period in ns to ps resolution (i.e. 33.333 is 30 MHz).
    -- CLKOUT0_DIVIDE - CLKOUT6_DIVIDE: Divide amount for each CLKOUT (1-128)
    CLKOUT1_DIVIDE => 25,
    --CLKOUT2_DIVIDE => 25,
    --CLKOUT3_DIVIDE => 1,
    --CLKOUT4_DIVIDE => 1,
    --CLKOUT5_DIVIDE => 1,
    --CLKOUT6_DIVIDE => 1,
    CLKOUT0_DIVIDE_F => 25.0, -- Divide amount for CLKOUT0 (1.000-128.000).
    -- CLKOUT0_DUTY_CYCLE - CLKOUT6_DUTY_CYCLE: Duty cycle for each CLKOUT (0.01-0.99).
    CLKOUT0_DUTY_CYCLE => 0.5,
    CLKOUT1_DUTY_CYCLE => 0.5,
    --CLKOUT2_DUTY_CYCLE => 0.5,
    --CLKOUT3_DUTY_CYCLE => 0.5,
    --CLKOUT4_DUTY_CYCLE => 0.5,
    --CLKOUT5_DUTY_CYCLE => 0.5,
    --CLKOUT6_DUTY_CYCLE => 0.5,
    -- CLKOUT0_PHASE - CLKOUT6_PHASE: Phase offset for each CLKOUT (-360.000-360.000).
    CLKOUT0_PHASE => 0.0,
    CLKOUT1_PHASE => 90.0,
    --CLKOUT2_PHASE => 0.0,
    --CLKOUT3_PHASE => 0.0,
    --CLKOUT4_PHASE => 0.0,
    --CLKOUT5_PHASE => 0.0,
    --CLKOUT6_PHASE => 0.0,
    CLKOUT4_CASCADE => FALSE, -- Cascade CLKOUT4 counter with CLKOUT6 (FALSE, TRUE)
    DIVCLK_DIVIDE => 1, -- Master division value (1-106)
    REF_JITTER1 => 0.0, -- Reference input jitter in UI (0.000-0.999).
    STARTUP_WAIT => FALSE -- Delays DONE until MMCM is locked (FALSE, TRUE)
    )
    port map (
    -- Clock Outputs: 1-bit (each) output: User configurable clock outputs
    CLKOUT0 => clk_out1_mmcm, -- 1-bit output: CLKOUT0
    --CLKOUT0B => CLKOUT0B, -- 1-bit output: Inverted CLKOUT0
    CLKOUT1 => clk_out2_mmcm, -- 1-bit output: CLKOUT1
    --CLKOUT1B => CLKOUT1B, -- 1-bit output: Inverted CLKOUT1
    --CLKOUT2 => CLKOUT2, -- 1-bit output: CLKOUT2
    --CLKOUT2B => CLKOUT2B, -- 1-bit output: Inverted CLKOUT2
    --CLKOUT3 => CLKOUT3, -- 1-bit output: CLKOUT3
    --CLKOUT3B => CLKOUT3B, -- 1-bit output: Inverted CLKOUT3
    --CLKOUT4 => CLKOUT4, -- 1-bit output: CLKOUT4
    --CLKOUT5 => CLKOUT5, -- 1-bit output: CLKOUT5
    --CLKOUT6 => CLKOUT6, -- 1-bit output: CLKOUT6
    -- Feedback Clocks: 1-bit (each) output: Clock feedback ports
    CLKFBOUT => clk_fout, -- 1-bit output: Feedback clock
    --CLKFBOUTB => CLKFBOUTB, -- 1-bit output: Inverted CLKFBOUT
    -- Status Ports: 1-bit (each) output: MMCM status ports
    LOCKED => lkd, -- 1-bit output: LOCK
    -- Clock Inputs: 1-bit (each) input: Clock input
    CLKIN1 => clk_in1, -- 1-bit input: Clock
    -- Control Ports: 1-bit (each) input: MMCM control ports
    PWRDWN => '0', -- 1-bit input: Power-down
    RST => reset, -- 1-bit input: Reset
    -- Feedback Clocks: 1-bit (each) input: Clock feedback ports
    CLKFBIN => clk_fout_buff -- 1-bit input: Feedback clock
    );
    -- End of MMCME2_BASE_inst instantiation  
    
    
    BUFG_inst : BUFG    --clock feedback buffer
    port map (
    O => clk_fout_buff, -- 1-bit output: Clock output
    I => clk_fout -- 1-bit input: Clock input
    );
    
    
    BUFGCE_clk_out1 : BUFGCE    --clock enable
    port map (
    O => clk_out1, -- 1-bit output: Clock output
    CE => clk1_en, -- 1-bit input: Clock enable input for I0
    I => clk_out1_mmcm -- 1-bit input: Primary clock
    );
    process (clk_out1_mmcm ) is --safe start up
    begin
        if rising_edge(clk_out1_mmcm) then
            clk1_en <= lkd ;    --load the clock enable signal
        end if;     
    end process;
    
    BUFGCE_clk_out2 : BUFGCE    --clock enable
    port map (
    O => clk_out2, -- 1-bit output: Clock output
    CE => clk2_en, -- 1-bit input: Clock enable input for I0
    I => clk_out2_mmcm -- 1-bit input: Primary clock
    );
    process (clk_out2_mmcm) is  --safe start up
    begin
        if rising_edge(clk_out2_mmcm) then
            clk2_en <= lkd ;    --load the clock enable signal
        end if;     
    end process;
    
    
    locked <= lkd;
    reset <= not rstn;
    
end Behavioral;