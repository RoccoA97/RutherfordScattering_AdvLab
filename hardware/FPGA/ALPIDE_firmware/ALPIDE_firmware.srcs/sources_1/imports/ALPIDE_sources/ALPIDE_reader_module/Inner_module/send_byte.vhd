----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 10.12.2019 14:59:51
-- Design Name: 
-- Module Name: send_byte - Behavioral
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
use IEEE.STD_LOGIC_ARITH.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity send_byte is
    Port ( clk_in : in STD_LOGIC;
           strt : in STD_LOGIC;    --write enable??
           pin_out : out STD_LOGIC; 
           ready : out STD_LOGIC;   --ready to eand
           word_sent : out STD_LOGIC; 
           word_in : in STD_LOGIC_VECTOR ( 7 downto 0)
           );
end send_byte;

architecture Behavioral of send_byte is

signal sending : STD_LOGIC;
signal word_buff : STD_LOGIC_VECTOR (7 downto 0) ;
type state is (s_idle, s_low, s_send, s_high);	--define state type 
signal state_fsm : state;

begin
    
    process (strt, clk_in)
        
    begin
        if rising_edge(clk_in) and strt = '1' and state_fsm = s_idle then
            word_buff <= word_in ;    
        end if;
    end process;
    
    send : process (clk_in, strt) is
    
    variable word_cnt : natural:= 0;
    
    begin
        if rising_edge(clk_in) then
            case state_fsm is
            when s_idle =>
                ready <= '1' ;
                word_sent <= '0' ;
                if strt = '1' then --if start button pressed
                    pin_out <= '1' ; 
                    state_fsm <= s_low ;
                else 
                    pin_out <= '1' ;
                    state_fsm <= s_idle ; --otherwise remains in idle
            end if;
            when s_low =>
                ready <= '0' ;
                word_sent <= '0' ;
                pin_out <= '0' ;
                state_fsm <= s_send ;
                when s_send =>
                    word_sent <= '0' ;		
                    pin_out <= word_buff(word_cnt) ;
                    if word_cnt >= 7 then --whole word sent!
                        state_fsm <= s_high ;	--change state
                        word_cnt := 0 ; --reset counter word
                    else 
                        word_cnt := word_cnt + 1 ;
                    end if;
                when s_high =>
                    word_sent <= '1' ;		
                    pin_out <= '1' ;
                    state_fsm <= s_idle ;
                    ready <= '0' ;
                when others =>
                    state_fsm <= s_idle ;
                    word_sent <= '0' ;
                    ready <= '0' ;
                end case;
            else 
                null;    
        end if;
    
    end process;

end Behavioral;
