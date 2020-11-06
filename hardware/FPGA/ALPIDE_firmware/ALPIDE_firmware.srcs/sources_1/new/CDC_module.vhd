library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

Library xpm;
use xpm.vcomponents.all;

entity CDC_module  is
    Generic (IO_DATA_WIDTH : integer := 1;
            sync_FF : integer := 2 );
    Port ( I_s : in STD_LOGIC_VECTOR (IO_DATA_WIDTH-1 downto 0);    --Input signal
           I_Clk : in STD_LOGIC;    --Source_clk
           O_Clk : in STD_LOGIC;    --Destination clk
           O_s : out STD_LOGIC_VECTOR (IO_DATA_WIDTH-1 downto 0));  --Output signal (sync)
end CDC_module;

architecture Behavioral of CDC_module is

begin
    
    xpm_cdc_array_single_inst : xpm_cdc_array_single
    generic map (
    DEST_SYNC_FF => sync_FF,      -- DECIMAL; range: 2-10
    INIT_SYNC_FF => 0,      -- DECIMAL; 0=disable simulation init values, 1=enable simulation init values
    SIM_ASSERT_CHK => 0,    -- DECIMAL; 0=disable simulation messages, 1=enable simulation messages
    SRC_INPUT_REG => 1,     -- DECIMAL; 0=do not register input, 1=register input
    WIDTH => IO_DATA_WIDTH  -- DECIMAL; range: 1-1024
    )
    port map (
    dest_out => O_s,    -- WIDTH-bit output: src_in synchronized to the destination clock domain. This
                        -- output is registered.
    dest_clk => O_Clk,  -- 1-bit input: Clock signal for the destination clock domain.
    src_clk => I_Clk,    -- 1-bit input: optional; required when SRC_INPUT_REG = 1
    src_in => I_s       --WIDTH-bit input: Input single-bit array to be synchronized to destination clock
                        --domain. It is assumed that each bit of the array is unrelated to the others.
                        --This is reflected in the constraints applied to this macro. To transfer a binary
                        --value losslessly across the two clock domains, use the XPM_CDC_GRAY macro
                        -- instead.
    );

end Behavioral;

