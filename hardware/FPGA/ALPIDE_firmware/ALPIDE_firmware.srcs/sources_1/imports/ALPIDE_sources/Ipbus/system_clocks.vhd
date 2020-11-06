-- clocks_v6
--
-- Generates a 125MHz ethernet clock and 31MHz ipbus clock from the 200MHz reference
-- Includes reset logic for ipbus
--
-- Dave Newbold, April 2011
-- INFN - LNL, December 2016
-- $Id$

library ieee;
use ieee.std_logic_1164.all;

library unisim;
use unisim.VComponents.all;

entity system_clocks is port(
  sysclk_p           : in  std_logic;
  clko_ipb           : out std_logic;
  sysclk_o           : out std_logic;
  sysclk_x2_o        : out std_logic;
  sysclk_x4_o        : out std_logic;
  phy_clk_o          : out std_logic;
  clk_200_o          : out std_logic;
  locked             : out std_logic;
  nuke               : in  std_logic;
  rsto_125           : out std_logic;
  rsto_ipb           : out std_logic;
  onehz              : out std_logic
  );

end system_clocks;

architecture rtl of system_clocks is
  
  signal dcm_locked, clk_ipb_i, clk_ipb_b : std_logic;
  signal d17, d17_d                       : std_logic;
  signal nuke_i, nuke_d, nuke_d2          : std_logic := '0';
  signal rst, rst_ipb, rst_125            : std_logic := '1';
  signal s_sysclk_x4_in                   : std_logic;
  signal s_fb_txclk_in                    : std_logic;
  signal s_fb_txclk_out                   : std_logic;
  signal s_sysclk_in                      : std_logic;
  signal s_sysclk_x2_in                   : std_logic;
  signal s_phy_clk_in                     : std_logic;
  signal s_sysclk_x2_b                    : std_logic;
  signal s_clk_base                       : std_logic;  -- 62.5 MHz onboard clock 
  signal s_clk_200_in                     : std_logic;
  
begin

--   BUFG_CLK_BASE : BUFG
--   port map (
--      O => s_clk_base,   -- 62.5 MHz
--      I => sysclk_p 
--   );
	
   s_clk_base <= sysclk_p;	
 	
   BUFG_SYS_CLK : BUFG
   port map (
      O => sysclk_o,      -- system clock 62.5 MHz
      I => s_sysclk_in
   );	

   BUFG_SYS_CLK_x4 : BUFG
   port map (
      O => sysclk_x4_o,   -- 250 MHz
      I => s_sysclk_x4_in 
   );

   BUFG_SYS_CLK_x2 : BUFG
   port map (
      O => s_sysclk_x2_b,    -- 125 MHz
      I => s_sysclk_x2_in 
   );
   sysclk_x2_o <= s_sysclk_x2_b;
	
	BUFG_TDC_CLK : BUFG
   port map (
      O => phy_clk_o,   -- 25 MHz
      I => s_phy_clk_in 
   );
	
   BUFG_SYS_CLK_FB : BUFG
   port map (
      O => s_fb_txclk_out,    -- fb TX PLL
      I => s_fb_txclk_in
   );
  
   bufgipb : BUFG port map(
    i => clk_ipb_i,
    o => clk_ipb_b
    );
	
	buf200 : BUFG port map(
    i => s_clk_200_in,
    o => clk_200_o
    );
	 

   clko_ipb <= clk_ipb_b;

  
   MMCME2_ADV_TX_PLL : MMCME2_ADV
   generic map (
      BANDWIDTH => "OPTIMIZED",      -- Jitter programming (OPTIMIZED, HIGH, LOW)
      CLKFBOUT_MULT_F => 20.000,        -- Multiply value for all CLKOUT (2.000-64.000).
      CLKFBOUT_PHASE => 0.0,         -- Phase offset in degrees of CLKFB (-360.000-360.000).
      -- CLKIN_PERIOD: Input clock period in ns to ps resolution (i.e. 33.333 is 30 MHz).
      CLKIN1_PERIOD => 10.0,
      CLKIN2_PERIOD => 0.0,
      -- CLKOUT0_DIVIDE - CLKOUT6_DIVIDE: Divide amount for CLKOUT (1-128)
      CLKOUT1_DIVIDE => 8,
      CLKOUT2_DIVIDE => 4,
      CLKOUT3_DIVIDE => 32,
      CLKOUT4_DIVIDE => 40,
      CLKOUT5_DIVIDE => 5,
      CLKOUT6_DIVIDE => 1,
      CLKOUT0_DIVIDE_F => 16.000,       -- Divide amount for CLKOUT0 (1.000-128.000).
      -- CLKOUT0_DUTY_CYCLE - CLKOUT6_DUTY_CYCLE: Duty cycle for CLKOUT outputs (0.01-0.99).
      CLKOUT0_DUTY_CYCLE => 0.5,
      CLKOUT1_DUTY_CYCLE => 0.5,
      CLKOUT2_DUTY_CYCLE => 0.5,
      CLKOUT3_DUTY_CYCLE => 0.5,
      CLKOUT4_DUTY_CYCLE => 0.5,
      CLKOUT5_DUTY_CYCLE => 0.5,
      CLKOUT6_DUTY_CYCLE => 0.5,
      -- CLKOUT0_PHASE - CLKOUT6_PHASE: Phase offset for CLKOUT outputs (-360.000-360.000).
      CLKOUT0_PHASE => 0.0,
      CLKOUT1_PHASE => 0.0,
      CLKOUT2_PHASE => 0.0,
      CLKOUT3_PHASE => 0.0,
      CLKOUT4_PHASE => 0.0,
      CLKOUT5_PHASE => 0.0,
      CLKOUT6_PHASE => 0.0,
      CLKOUT4_CASCADE => FALSE,      -- Cascade CLKOUT4 counter with CLKOUT6 (FALSE, TRUE)
      COMPENSATION => "ZHOLD",       -- ZHOLD, BUF_IN, EXTERNAL, INTERNAL
      DIVCLK_DIVIDE => 2,            -- Master division value (1-106)
      -- REF_JITTER: Reference input jitter in UI (0.000-0.999).
      REF_JITTER1 => 0.0,
      REF_JITTER2 => 0.0,
      STARTUP_WAIT => FALSE,         -- Delays DONE until MMCM is locked (FALSE, TRUE)
      -- Spread Spectrum: Spread Spectrum Attributes
      SS_EN => "FALSE",              -- Enables spread spectrum (FALSE, TRUE)
      SS_MODE => "CENTER_HIGH",      -- CENTER_HIGH, CENTER_LOW, DOWN_HIGH, DOWN_LOW
      SS_MOD_PERIOD => 10000,        -- Spread spectrum modulation period (ns) (VALUES)
      -- USE_FINE_PS: Fine phase shift enable (TRUE/FALSE)
      CLKFBOUT_USE_FINE_PS => FALSE,
      CLKOUT0_USE_FINE_PS => FALSE,
      CLKOUT1_USE_FINE_PS => FALSE,
      CLKOUT2_USE_FINE_PS => FALSE,
      CLKOUT3_USE_FINE_PS => FALSE,
      CLKOUT4_USE_FINE_PS => FALSE,
      CLKOUT5_USE_FINE_PS => FALSE,
      CLKOUT6_USE_FINE_PS => FALSE 
   )
   port map (
      -- Clock Outputs: 1-bit (each) output: User configurable clock outputs
      CLKOUT0 => s_sysclk_in,                  -- system clock 62.5 MHz
      CLKOUT0B => open,     
      CLKOUT1 => s_sysclk_x2_in,               -- system clock 125 MHz
      CLKOUT1B => open,      
      CLKOUT2 => s_sysclk_x4_in,               -- system clock 250 MHz
      CLKOUT2B => open,      
      CLKOUT3 => clk_ipb_i,                    -- system clock 31.25 MHz
      CLKOUT3B => open,         
      CLKOUT4 => s_phy_clk_in,                 -- system clock 25 MHz
      CLKOUT5 => s_clk_200_in,                 -- system clock 200 MHz
      CLKOUT6 => open,          
      -- DRP Ports: 16-bit (each) output: Dynamic reconfiguration ports
      DO => open,                    
      DRDY => open,                
      -- Dynamic Phase Shift Ports: 1-bit (each) output: Ports used for dynamic phase shifting of the outputs
      PSDONE => open,            
      -- Feedback Clocks: 1-bit (each) output: Clock feedback ports
      CLKFBOUT => s_fb_txclk_in,        
      CLKFBOUTB => open,       
      -- Status Ports: 1-bit (each) output: MMCM status ports
      CLKFBSTOPPED => open, 
      CLKINSTOPPED => open, 
      LOCKED => dcm_locked,             
      -- Clock Inputs: 1-bit (each) input: Clock inputs
      CLKIN1 => s_clk_base,             
      CLKIN2 => '0',             
      -- Control Ports: 1-bit (each) input: MMCM control ports
      CLKINSEL => '1',         
      PWRDWN => '0',           
      RST => '0',                
      -- DRP Ports: 7-bit (each) input: Dynamic reconfiguration ports
      DADDR => (others => '0'),               
      DCLK => '0',                 
      DEN => '0',                   
      DI => (others => '0'),                     
      DWE => '0',                   
      -- Dynamic Phase Shift Ports: 1-bit (each) input: Ports used for dynamic phase shifting of the outputs
      PSCLK => '0',               -- 1-bit input: Phase shift clock
      PSEN => '0',                 -- 1-bit input: Phase shift enable
      PSINCDEC => '0',         -- 1-bit input: Phase shift increment/decrement
      -- Feedback Clocks: 1-bit (each) input: Clock feedback ports
      CLKFBIN => s_fb_txclk_out            -- 1-bit input: Feedback clock
   );
  

  clkdiv : entity work.clock_div port map(
    clk => sysclk_p,
    d17 => d17,
    d28 => onehz
    );

  process(sysclk_p)
  begin
    if rising_edge(sysclk_p) then
      d17_d <= d17;
      if d17 = '1' and d17_d = '0' then
        rst     <= nuke_d2 or not dcm_locked;
        nuke_d  <= nuke_i;  -- Time bomb (allows return packet to be sent)
        nuke_d2 <= nuke_d;
      end if;
    end if;
  end process;

  locked <= dcm_locked;

  process(clk_ipb_b)
  begin
    if rising_edge(clk_ipb_b) then
      rst_ipb <= rst;
      nuke_i  <= nuke;
    end if;
  end process;

  rsto_ipb <= rst_ipb;

  process(s_sysclk_x2_b)
  begin
    if rising_edge(s_sysclk_x2_b) then
      rst_125 <= rst;
    end if;
  end process;

  rsto_125 <= rst_125;
  
end rtl;

