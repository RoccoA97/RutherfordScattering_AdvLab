----------------------------------------------------------------------------------
-- Company: 
-- Student: Gabriele Bortolato
-- 
-- Create Date: 10.12.2019 14:23:50
-- Design Name: ALPIDE on FPGA 
-- Module Name: ALPIDE_Carrier - Behavioral 
-- Target Devices: Arty A7 
-- Description: 
-- 

-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

library UNISIM;
use UNISIM.VComponents.all;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;


entity ALPIDE_Carrier is
    Generic (MAX_SLV_CLK : STD_LOGIC_VECTOR(7 downto 0) := x"33");
        
    Port ( clk_100MHz : in STD_LOGIC;   --sys clock 100MHz
    
           CTRL : inout STD_LOGIC;  --data pin
           DCLK : out STD_LOGIC;    --MASTER clk
           pwr_enable : out STD_LOGIC ; --ALPIDE power enable
           rst_n_io : out STD_LOGIC := '1' ;    --ALPIDE resetn 
           
           op_sw : in STD_LOGIC_VECTOR (3 downto 0) := "0000";    --16 selectable options
           A_strt : in STD_LOGIC;  --validate option and start selected routine
           rstn : in STD_LOGIC := '1';  --Arty resetn
           
           RR_addr : in STD_LOGIC_VECTOR (15 downto 0) ; 
           OP_command : in STD_LOGIC_VECTOR (7 downto 0) ; 
           CHIP_id : in STD_LOGIC_VECTOR (7 downto 0) ;   --Chip identifier (MULTICAST ID is 0F) set via ALPIDE switches
           WR_addr : in STD_LOGIC_VECTOR (15 downto 0) ;
           WR_data : in STD_LOGIC_VECTOR (15 downto 0) ;
           
           data_reg : out STD_LOGIC_VECTOR (15 downto 0) := x"FAFA" ; --data read on selected register
           data_FIFO : out STD_LOGIC_VECTOR (31 downto 0) := x"F0F0F0F0" ;   --data read on FIFO
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
end ALPIDE_Carrier;

architecture Behavioral of ALPIDE_Carrier is

component RW_PLL
    Port (clk_in1 : in STD_LOGIC;   --sys clock 100MHz
          rstn : in STD_LOGIC;      --sys resetn
          clk_out1 : out STD_LOGIC; --40MHz base clock
          clk_out2 : out STD_LOGIC;  --90� phase shifted 40MHz clock
          locked : out STD_LOGIC   --locked signal
          );
end component;           
    
signal m_clk_ph, m_clk : STD_LOGIC;  

--component CMD_ROM   --stored CMD for initialization
    --PORT (addra : in STD_LOGIC_VECTOR ( 7 downto 0 );
          --clka : in STD_LOGIC; 
          --douta : out STD_LOGIC_VECTOR ( 7 downto 0 )
          --);
--end component;

component ROM_INIT --stored CMD for initialization
    PORT (addra : in STD_LOGIC_VECTOR ( 7 downto 0 );
          clka : in STD_LOGIC; 
          ena : in STD_LOGIC;
          douta : out STD_LOGIC_VECTOR ( 7 downto 0 )
          );
end component;

signal addra, douta : STD_LOGIC_VECTOR ( 7 downto 0 ) ; --need 2 clk cycle to be effective 

component read_byte 
    Port ( clk_in : in STD_LOGIC;
           clk_ph : in STD_LOGIC;
           rstn : in STD_LOGIC;
           strt : in STD_LOGIC;
           pin_in : in STD_LOGIC;
           error, busy, readable : out STD_LOGIC;   
           word_out : out STD_LOGIC_VECTOR (7 downto 0) 
           );
end component;

signal re, error, busy, readable, pin_in : STD_LOGIC;
signal word_out, word_buff : STD_LOGIC_VECTOR (7 downto 0);


component send_byte 
    Port ( clk_in : in STD_LOGIC;
           strt: in STD_LOGIC;
           pin_out : out STD_LOGIC; 
           ready : out STD_LOGIC;
           word_sent : out STD_LOGIC;
           word_in : in STD_LOGIC_VECTOR ( 7 downto 0)
           );
end component;

signal we, word_sent, ready_to_send : STD_LOGIC;
signal pin_out : STD_LOGIC;
signal word_in : STD_LOGIC_VECTOR ( 7 downto 0);

signal A_strt1, A_strt2 : STD_LOGIC := '0' ;  --to detect rising or falling edge of A_strt

type task is (reading, sending, idling_m, idling_s);
signal current_task : task := idling_m ;

type state is (s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10);
signal i_state : state;
signal rr_state : state;
signal ro_state : state;
signal w_state : state;
signal wr_state : state;
type cmd_state is (init, rst_err, power_on, power_off, readregister, read_out, idle, writecommand, writeregister, trigger); 
signal ALPIDEC_state : cmd_state := idle;
signal initialized, powered, r_read, ro_done, c_write : STD_LOGIC; 

signal RR_command : STD_LOGIC_VECTOR (7 downto 0) := x"4E" ; --UNICAST read
signal WR_command : STD_LOGIC_VECTOR (7 downto 0) := x"9C" ; --UNICAST or MULTICAST write
signal OP_command_trigger : STD_LOGIC_VECTOR (7 downto 0) := x"55" ;



begin

    Read_FSM : read_byte PORT MAP 
    (clk_in => m_clk,
    clk_ph => m_clk_ph,
    rstn => rstn,
    strt => re, --read enable
    pin_in => pin_in,
    error => error,
    busy=> busy,
    readable => readable ,
    word_out => word_out);
    
    Write_FSM : send_byte PORT MAP  
    (clk_in => m_clk,
    strt => we, --write enable
    pin_out => pin_out,
    ready => ready_to_send,
    word_sent=> word_sent,
    word_in => word_in) ;
   
    MMCM : RW_PLL PORT MAP
    (clk_in1 => clk_100MHz,
    rstn => rstn,
    clk_out1 => m_clk,
    clk_out2 => m_clk_ph,
    locked => lkd) ;
    
    ROM : ROM_INIT PORT MAP
    (addra => addra,
    clka => m_clk,
    ena => '1',
    douta => douta) ;
    
    
    process(m_clk,rstn) is
    begin
    
        if rstn = '0' then
            A_strt2 <= '0' ;
            A_strt1 <= '0' ;
        elsif rising_edge(m_clk) then
            A_strt2 <= A_strt1 ;
            A_strt1 <= A_strt ;
        end if;
            
    end process ; 
      
    ---------------------------------
    --Command console
    ---------------------------------
    cmd_c : process(m_clk, A_strt1, A_strt2, rstn) is
    
    begin
        if rstn = '0' then
            ALPIDEC_state <= idle;
        elsif rising_edge(m_clk) then   
            case ALPIDEC_state is
            when idle =>
                busy_led <= '0' ;
                if A_strt1= '1' and A_strt2 = '0' then
                    case op_sw is
                    when x"0" =>
                        ALPIDEC_state <= power_off ;
                    when x"1" =>
                        ALPIDEC_state <= power_on ;
                    when x"2" =>
                        if initialized = '0' and powered = '1' then
                            ALPIDEC_state <= init ;
                        end if;
                    when x"4" =>
                        if initialized = '1' and powered = '1' then
                            ALPIDEC_state <= readregister ; --op command, chip_id, addrl, addrh, bus turnaround, chip_id, datal, datah, busturnaround
                        end if;
                    when x"5" =>
                        if initialized = '1' and powered = '1' then
                            ALPIDEC_state <= read_out ; --40 reading from FIFO
                        end if;
                    when x"8" =>
                        if initialized = '1' and powered = '1' then
                            ALPIDEC_state <= writecommand ; --op command(command)    
                        end if;
                    when x"9" =>
                        if initialized = '1' and powered = '1' then
                            ALPIDEC_state <= writeregister ;    --op command, chip_id (or multicast_id), addrl, addrh, datal, datah
                        end if;
                    when x"A" =>
                        if initialized = '1' and powered = '1' then
                            ALPIDEC_state <= trigger ;    --op command x"55"
                        end if;
                    when x"F" =>
                        ALPIDEC_state <= rst_err ;  --reset all error  
                    when others =>
                        null;
                    end case;
                end if;
            when rst_err =>
                ALPIDEC_state <= idle ;
            when init =>
                busy_led <= '1' ;
                if initialized = '1' then
                    ALPIDEC_state <= idle ;
                end if; 
            when power_on =>
                busy_led <= '1' ;
                if powered = '1' then 
                    ALPIDEC_state <= idle ;   
                end if;
            when power_off =>
                busy_led <= '1' ;
                if powered = '0' then
                    ALPIDEC_state <= idle ;          
                end if;
                ALPIDEC_state <= idle ;
            when readregister =>
                busy_led <= '1' ;
                if r_read = '1' then
                     ALPIDEC_state <= idle ;
                end if;
            when read_out =>
                busy_led <= '1' ;
                if ro_done = '1' then
                     ALPIDEC_state <= idle ;
                end if;
            when writecommand =>
                busy_led <= '1' ;
                if c_write = '1' then
                     ALPIDEC_state <= idle ;
                end if;
            when writeregister =>
                busy_led <= '1' ;
                if c_write = '1' then
                     ALPIDEC_state <= idle ;
                end if;
            when trigger =>
                busy_led <= '1' ;
                if c_write = '1' then
                     ALPIDEC_state <= idle ;
                end if;     
            when others =>
                null; 
            end case;    
        end if;
    end process cmd_c;
    
    
    ---------------------------------
    --CTRL inout port
    ------------------------------------
    process(current_task, pin_in, pin_out, CTRL) is
    
    begin
        case current_task is
        when sending =>
            CTRL <= pin_out ;
        when reading =>
            CTRL <= 'Z'  ; 
            pin_in <= CTRL ;
        when idling_m =>    
            CTRL <= '1' ;   --idle driven by master (FPGA)    
        when idling_s =>
            CTRL <= 'Z' ;   --idle driven by slave (ALPIDE chip)
        when others =>
            CTRL <= 'Z' ;   --set high impedace just for safety  
        end case;                   
        
    end process;
    ---------------------------------------
    
    main : process (m_clk,rstn) is
    
    variable cmd_cnt : STD_LOGIC_VECTOR( 7 downto 0 ) := x"00" ;
    variable byte_cnt, idle_cnt, slave_cnt : STD_LOGIC_VECTOR( 7 downto 0 ) := x"00" ;
    variable FIFO_cnt : STD_LOGIC_VECTOR( 1 downto 0 ) := "00" ;
    variable data_cnt : STD_LOGIC_VECTOR( 11 downto 0 ) := x"000" ;
    
    begin
    
        if rstn = '0' then
            current_task <= idling_s ;  --to prevent damage on ALPIDE, if it is sending
            i_state <= s0 ; --starting state
            rr_state <= s0 ;
            initialized <= '0' ;
            d_reg_we <= '0' ;
            d_FIFO_we <= '0' ;
            r_read <= '0' ;
            ro_done <= '0' ;
            idle_err <= '0' ;
            read_err<= '0' ;
            chip_id_err <= '0' ;
            slave_err <= '0' ;
            c_write <= '0' ;
            cmd_cnt := x"00" ;
            byte_cnt := x"00" ;
            idle_cnt := x"00" ;
            FIFO_CNT := "00" ;
            data_cnt := x"000" ;
        elsif rising_edge(m_clk) then
            case ALPIDEC_state is
            when idle =>
                if powered = '1' then
                    current_task <= idling_m ;
                else 
                    current_task <= idling_s ;
                end if;
                r_read <= '0' ;
                c_write <= '0' ;
                ro_done <= '0' ;
            when power_off =>   --power off routine
                if powered = '1' then
                    rst_n_io <= '0' ; 
                    pwr_enable <= '0' ;
                    powered <= '0' ;
                    initialized <= '0' ;          
                end if;
            when power_on =>    --power on routine
                if powered = '1' then
                    rst_n_io <= '1' ;
                    pwr_enable <= '1' ; 
                else 
                    rst_n_io <= '0' ;
                    pwr_enable <= '0' ;
                    powered <= '1' ;    
                end if;
            when rst_err=>
                idle_err<= '0' ;
                read_err <= '0' ;
                chip_id_err <= '0' ;
                slave_err <= '0' ;    
            when init =>    --initialization routine
                case i_state is
                when s0 => 
                    if ALPIDEC_state = init and initialized = '0' then
                        addra <= x"00" ; --initialize to base address
                        i_state <= s1 ;
                        rst_n_io <= '1' ;
                    else
                        i_state <= s0 ;
                    end if;
                when s1 =>
                    rst_n_io <= '0' ;
                    i_state <= s2 ;   
                when s2 =>
                    rst_n_io <= '1' ;
                    i_state <= s3 ;
                when s3 =>                
                    current_task <= sending ;
                    if ready_to_send = '1' then
                        word_in <= douta ;   --from ROM
                        we <= '1' ;
                        i_state <= s4 ;
                     else 
                        i_state <= s3 ;
                     end if;
                when s4 =>
                    we <= '0' ;
                    addra <= addra + x"01" ;
                    i_state <= s5 ;
                when s5 =>
                    if word_sent = '1' then
                        if cmd_cnt >= x"38" then
                            addra <= x"00" ; --to base ROM address
                            cmd_cnt := x"00" ;
                            initialized <= '1' ;
                            i_state <= s0 ;
                            current_task <= idling_m ;
                        else 
                            cmd_cnt := cmd_cnt + x"01" ;
                            i_state <= s3 ;   
                        end if;
                    end if;
                when others =>
                    i_state <= s0 ;
                end case;
            when readregister =>    --readregister routine
                case rr_state is
                when s0 => 
                    d_reg_we <= '0' ;
                    if ALPIDEC_state = readregister and r_read = '0' then
                        rr_state <= s1 ;
                    else
                        rr_state <= s0 ;
                    end if;
                when s1 =>
                    current_task <= sending ;
                    if ready_to_send = '1' then
                        we <= '1' ;
                        case byte_cnt is
                        when x"00" =>
                            word_in <= RR_command ; --UNICAST read  
                        when x"01" =>
                            word_in <= CHIP_id ; --chip id
                        when x"02" =>
                            word_in <= RR_addr( 7 downto 0 ) ;   --register address    
                        when x"03" =>
                            word_in <= RR_addr( 15 downto 8 ) ;
                        when others => 
                            null; 
                        end case;
                        rr_state <= s2 ;
                     else 
                        rr_state <= s1 ;
                     end if;
                when s2 =>
                    we <= '0' ;
                    rr_state <= s3 ;
                when s3 =>
                    if word_sent = '1' then
                        if byte_cnt >= x"03" then
                            byte_cnt := x"00" ;
                            current_task <= idling_m ;
                            rr_state <= s4 ;
                        else
                            byte_cnt := byte_cnt + x"01" ;
                            rr_state <= s1 ;        
                        end if;
                    end if;
                when s4 =>  --idle_m and idle_s phase, bus turnaround
                    if idle_cnt < x"04" then
                        current_task <= idling_m ;  --master driver on
                        idle_cnt := idle_cnt + x"01" ;
                    elsif  idle_cnt >= x"04" and idle_cnt < x"08" then 
                        current_task <= idling_s ;  --turnaround
                        slave_cnt := slave_cnt + x"01" ;
                        idle_cnt := idle_cnt + x"01" ;
                    elsif  idle_cnt >= x"08" and idle_cnt < x"0A" then 
                        current_task <= reading ;  --slave driver on
                        slave_cnt := slave_cnt + x"01" ;
                        idle_cnt := idle_cnt + x"01" ;
                        if CTRL /= '1' then
                           idle_err <= '1' ;
                        end if;
                    else    
                        current_task <= reading ;  --slave driver on
                        slave_cnt := slave_cnt + x"01" ;
                        idle_cnt := x"00" ;
                        if CTRL /= '1' then
                            idle_err <= '1' ;
                        end if;
                        rr_state <= s5 ;
                    end if;
                when s5 =>
                    slave_cnt := slave_cnt + x"01" ;
                    current_task <= reading ;
                    re <= '1' ;
                    if CTRL /= '1' then
                            idle_err <= '1' ;
                        end if;
                    rr_state <= s6 ;  
                when s6 =>
                    slave_cnt := slave_cnt + x"01" ;
                    if busy = '1' then
                        re <= '0' ;
                        rr_state <= s7 ;
                    else
                        re <= '1' ;
                        rr_state <= s6 ;  
                    end if;
                    if slave_cnt > MAX_SLV_CLK then
                        slave_err <= '1' ;
                        slave_cnt := x"00" ;
                        r_read <= '1' ;
                        current_task <= idling_m ;
                        rr_state <= s0 ;     
                    end if;
                when s7 => 
                    slave_cnt := slave_cnt + x"01" ;
                    if readable = '1' then
                        case byte_cnt is
                        when x"00" =>
                            if word_out = CHIP_id then
                                chip_id_err <= '0' ;
                            else 
                                chip_id_err <= '1' ;
                            end if; 
                            byte_cnt := byte_cnt + x"01" ;
                            re <= '1' ;
                            rr_state <= s6 ;  
                        when x"01" =>
                            data_reg( 7 downto 0 ) <= word_out ;    --load data low
                            byte_cnt := byte_cnt + x"01" ;
                            re <= '1' ;
                            rr_state <= s6 ;
                        when x"02" =>
                            data_reg( 15 downto 8 ) <= word_out ;   --load data high
                            d_reg_we <= '1' ;   --data can be stored in bram  
                            byte_cnt := x"00" ;
                            rr_state <= s8 ;
                        when others =>
                            byte_cnt := x"00" ;
                            chip_id_err <= '1' ;
                            rr_state <= s8 ;
                        end case;
                    elsif error = '1' then
                        slave_cnt := x"00" ;
                        read_err <= '1' ;  
                        rr_state <= s9 ;    --error state 
                    else
                        re <= '1' ;
                        rr_state <= s7 ;  
                    end if;
                    if slave_cnt > MAX_SLV_CLK then
                        slave_err <= '1' ;
                        slave_cnt := x"00" ;
                        r_read <= '1' ;
                        current_task <= idling_m ;
                        rr_state <= s0 ;     
                    end if;
                when s8 => --bus turnaround
                    slave_cnt := x"00" ;
                    if idle_cnt < x"07" then
                        current_task <= idling_s ;  --slave driver on and turnaround
                        idle_cnt := idle_cnt + x"01" ;
                    elsif  idle_cnt >= x"07" and idle_cnt <= x"0A" then 
                        current_task <= idling_m ;  --turnaround and master driver on
                        idle_cnt := idle_cnt + x"01" ;
                    else
                        idle_cnt := x"00" ;
                        r_read <= '1' ; --register read
                        rr_state <= s0 ;
                    end if;
                when s9 =>  --error state
                    d_reg_we <= '0' ;
                    slave_cnt := slave_cnt + x"01" ;
                    current_task <= idling_s ;
                    if slave_cnt > MAX_SLV_CLK then
                        slave_err <= '1' ;
                        slave_cnt := x"00" ;
                        r_read <= '1' ;
                        current_task <= idling_m ;  
                        rr_state <= s0 ;   
                    end if;                
                when others =>
                    rr_state <= s0 ;
                end case;
            when read_out =>
                if FIFO_cnt < "10" then
                    case ro_state is
                    when s0 => 
                        if ro_FIFO_full /= '1' then
                            d_FIFO_we <= '0' ;
                            if ALPIDEC_state = read_out and ro_done = '0' then
                                ro_state <= s1 ;
                            else
                                ro_state <= s0 ;
                            end if;
                        elsif ro_FIFO_full = '1' then
                            ro_state <= s0 ;
                        end if;
                    when s1 =>
                        current_task <= sending ;
                        if ready_to_send = '1' then
                            we <= '1' ;
                            case byte_cnt is
                            when x"00" =>
                                word_in <= RR_command ; --UNICAST read  
                            when x"01" =>
                                word_in <= CHIP_id ; --chip id
                            when x"02" =>
                                case FIFO_cnt is
                                when "00" =>
                                    word_in <= x"12" ;   
                                when "01" => 
                                    word_in <= x"13" ;
                                when others =>
                                    word_in <= x"00" ;
                                end case;       
                            when x"03" =>
                                word_in <= x"00" ;
                            when others => 
                                null; 
                            end case;
                           ro_state <= s2 ;
                        else 
                           ro_state <= s1 ;
                        end if;
                    when s2 =>
                        we <= '0' ;
                        ro_state <= s3 ;
                    when s3 =>
                        if word_sent = '1' then
                            if byte_cnt >= x"03" then
                                byte_cnt := x"00" ;
                                current_task <= idling_m ;
                                ro_state <= s4 ;
                            else
                                byte_cnt := byte_cnt + x"01" ;
                                ro_state <= s1 ;        
                            end if;
                        end if;
                    when s4 =>  --idle_m and idle_s phase, bus turnaround
                        if idle_cnt < x"04" then
                            current_task <= idling_m ;  --master driver on
                            idle_cnt := idle_cnt + x"01" ;
                        elsif  idle_cnt >= x"04" and idle_cnt < x"08" then 
                            current_task <= idling_s ;  --turnaround
                            slave_cnt := slave_cnt + x"01" ;
                            idle_cnt := idle_cnt + x"01" ;
                        elsif  idle_cnt >= x"08" and idle_cnt < x"0A" then 
                            current_task <= reading ;  --slave driver on
                            slave_cnt := slave_cnt + x"01" ;
                            idle_cnt := idle_cnt + x"01" ;
                            if CTRL /= '1' then
                                idle_err <= '1' ;
                            end if;
                        else    
                            current_task <= reading ;  --slave driver on
                            slave_cnt := slave_cnt + x"01" ;
                            idle_cnt := x"00" ;
                            if CTRL /= '1' then
                                idle_err <= '1' ;
                            end if;
                            ro_state <= s5 ;
                        end if;
                    when s5 =>
                        slave_cnt := slave_cnt + x"01" ;
                        current_task <= reading ;
                        re <= '1' ;
                        if CTRL /= '1' then
                                idle_err <= '1' ;
                            end if;
                        ro_state <= s6 ;  
                    when s6 =>
                        slave_cnt := slave_cnt + x"01" ;
                        if busy = '1' then
                            re <= '0' ;
                            ro_state <= s7 ;
                        else
                            re <= '1' ;
                            ro_state <= s6 ;  
                        end if;
                        if slave_cnt > MAX_SLV_CLK then
                            slave_err <= '1' ;
                            slave_cnt := x"00" ;
                            ro_done <= '1' ;
                            current_task <= idling_m ;
                            ro_state <= s0 ;     
                        end if;
                    when s7 => 
                        slave_cnt := slave_cnt + x"01" ;
                        if readable = '1' then
                            case byte_cnt is
                            when x"00" =>
                                if word_out = CHIP_id then
                                    chip_id_err <= '0' ;
                                else 
                                    chip_id_err <= '1' ;
                                end if; 
                                byte_cnt := byte_cnt + x"01" ;
                                re <= '1' ;
                                ro_state <= s6 ;  
                            when x"01" =>
                                case FIFO_cnt is
                                when "00" =>
                                    data_FIFO( 7 downto 0 ) <= word_out ;    
                                when "01" =>
                                    data_FIFO( 23 downto 16 ) <= word_out ;
                                when others =>
                                        --erorr maybe
                                    null;
                                end case;  
                                byte_cnt := byte_cnt + x"01" ;
                                re <= '1' ;
                                ro_state <= s6 ;
                            when x"02" =>
                                case FIFO_cnt is
                                when "00" =>
                                    data_FIFO( 15 downto 8 ) <= word_out ;    
                                when "01" =>
                                    data_FIFO( 31 downto 24 ) <= word_out ; --should be x"00"
                                    d_FIFO_we <= '1' ;   --data can be stored in bram
                                when others =>
                                        --erorr maybe
                                    null;
                                end case;
                                byte_cnt := x"00" ;
                                ro_state <= s8 ;
                            when others =>
                                byte_cnt := x"00" ;
                                chip_id_err <= '1' ;
                                ro_state <= s8 ;
                            end case;
                        elsif error = '1' then
                            slave_cnt := x"00" ;
                            read_err <= '1' ;  
                            ro_state <= s9 ;    --error state 
                        else
                            re <= '1' ;
                            ro_state <= s7 ;  
                        end if;
                        if slave_cnt > MAX_SLV_CLK then
                            slave_err <= '1' ;
                            slave_cnt := x"00" ;
                            ro_done <= '1' ;
                            current_task <= idling_m ;
                            ro_state <= s0 ;     
                        end if;
                    when s8 => --bus turnaround
                        slave_cnt := x"00" ;
                        if idle_cnt < x"07" then
                            current_task <= idling_s ;  --slave driver on and turnaround
                            idle_cnt := idle_cnt + x"01" ;
                        elsif  idle_cnt >= x"07" and idle_cnt <= x"0A" then 
                            current_task <= idling_m ;  --turnaround and master driver on
                            idle_cnt := idle_cnt + x"01" ;
                        else
                            idle_cnt := x"00" ;
                            FIFO_cnt := FIFO_cnt + "01" ;
                            ro_state <= s0 ;
                        end if;
                    when s9 =>  --error state
                        d_FIFO_we <= '0' ;
                        slave_cnt := slave_cnt + x"01" ;
                        current_task <= idling_s ;
                        if slave_cnt > MAX_SLV_CLK then
                            slave_err <= '1' ;
                            slave_cnt := x"00" ;
                            ro_done <= '1' ;
                            current_task <= idling_m ;  
                            ro_state <= s0 ;   
                        end if;                
                    when others =>
                        ro_state <= s0 ;
                    end case;
                else
                    FIFO_cnt := "00" ;
                    if ro_stop = '1' then 
                        ro_state <= s0 ;
                        ro_done <= '1' ;    
                    end if;       
                end if;    
            when writecommand =>
            case w_state is
                when s0 => 
                    if ALPIDEC_state = writecommand and c_write = '0' then
                        w_state <= s1 ;
                    else
                        w_state <= s0 ;
                    end if;
                when s1 =>
                    current_task <= sending ;
                    if ready_to_send = '1' then
                        word_in <= OP_command ;   
                        we <= '1' ;
                        w_state <= s2 ;
                     else 
                        w_state <= s1 ;
                     end if;   
                when s2 =>
                    we <= '0' ;
                    w_state <= s3 ;
                when s3 =>                
                    if word_sent = '1' then
                        w_state <= s0 ;
                        c_write <= '1' ;
                    end if;
                when others =>
                    w_state <= s0 ;
                end case;
            when writeregister =>
                case wr_state is
                when s0 => 
                    if c_write = '0' then
                        wr_state <= s1 ;
                    else
                        wr_state <= s0 ;
                    end if;
                when s1 =>
                    current_task <= sending ;
                    if ready_to_send = '1' then
                        we <= '1' ;
                        case byte_cnt is
                        when x"00" =>
                            word_in <= WR_command ; --UNICAST or MULTICAST write   
                        when x"01" =>
                            word_in <= CHIP_id ; --chip id or MULTICAST id
                        when x"02" =>
                            word_in <= WR_addr( 7 downto 0 ) ;   --register address    
                        when x"03" =>
                            word_in <= WR_addr( 15 downto 8 ) ;
                        when x"04" =>
                            word_in <= WR_data( 7 downto 0 ) ;    --data to write
                        when x"05" =>
                            word_in <= WR_data( 15 downto 8 ) ;
                        when others => 
                            null; 
                        end case;
                        wr_state <= s2 ;
                    else
                        wr_state <= s1 ;
                    end if;
                when s2 =>
                    we <= '0' ;
                    wr_state <= s3 ;
               when s3 =>
                    if word_sent = '1' then
                        if byte_cnt >= x"05" then
                            byte_cnt := x"00" ;
                            wr_state <= s0 ;
                            c_write <= '1' ;
                        else
                            byte_cnt := byte_cnt + x"01" ;
                            wr_state <= s1 ;
                        end if ;
                    end if;
                when others =>
                    wr_state <= s0 ;
                end case;
            when trigger =>
            case w_state is
                when s0 => 
                    if ALPIDEC_state = trigger and c_write = '0' then
                        w_state <= s1 ;
                    else
                        w_state <= s0 ;
                    end if;
                when s1 =>
                    current_task <= sending ;
                    if ready_to_send = '1' then
                        word_in <= OP_command_trigger ;   
                        we <= '1' ;
                        w_state <= s2 ;
                     else 
                        w_state <= s1 ;
                     end if;   
                when s2 =>
                    we <= '0' ;
                    w_state <= s3 ;
                when s3 =>                
                    if word_sent = '1' then
                        w_state <= s0 ;
                        c_write <= '1' ;
                    end if;
                when others =>
                    w_state <= s0 ;
                end case;  
            when others =>
                null ;    
            end case;
        end if;
    end process main;
    
    init_led <= initialized ;
    power_led <= powered ;
    DCLK <= m_clk ;  --clk to ALPIDE chip
    
end Behavioral;

------------------------------------------
--TODO
--add byte sequence (ROM) in init routine ------- DONE
--add write register and write command options ------- DONE
--check number of clk cycles on bus turnaround  -------DONE   
--don't know what to do with the data line when reset is pressed 
--add PWM RGB led controller for error led (stock ones are too strong)  ------DONE
--add reset function only for errors    --------DONE
--on power off need to lock CTRL to low --------DONE